import asyncio
import base64
import math
import os
import shutil
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer

DEFAULT_PROMPT = os.getenv(
    "DEEPSEEK_OCR_PROMPT",
    "<image>\n<|grounding|>Convert the document to markdown.",
)
MODEL_ID = os.getenv("DEEPSEEK_OCR_MODEL_ID", "deepseek-ai/DeepSeek-OCR")
BASE_SIZE = int(os.getenv("DEEPSEEK_OCR_BASE_SIZE", "1024"))
IMAGE_SIZE = int(os.getenv("DEEPSEEK_OCR_IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("DEEPSEEK_OCR_CROP_MODE", "true").lower() not in {"0", "false"}
IDLE_TIMEOUT_SECONDS = int(os.getenv("DEEPSEEK_OCR_IDLE_TIMEOUT", "1800"))
PDF_DPI = int(os.getenv("DEEPSEEK_OCR_PDF_DPI", "144"))
MAX_PDF_PAGES = int(os.getenv("DEEPSEEK_OCR_MAX_PAGES", "0"))
MIN_CROPS = int(os.getenv("DEEPSEEK_OCR_MIN_CROPS", "2"))
MAX_CROPS = int(os.getenv("DEEPSEEK_OCR_MAX_CROPS", "9"))
SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
CONTENT_TYPE_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


def _make_temp_file(data: bytes, suffix: str) -> str:
    hd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(hd, "wb") as fh:
            fh.write(data)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(path)
        raise
    return path


def _pdf_bytes_to_image_paths(data: bytes, dpi: int) -> List[str]:
    document = fitz.open(stream=data, filetype="pdf")
    image_paths: List[str] = []
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(document.page_count):
            pixmap = document[page_index].get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pixmap.tobytes("png")
            image_paths.append(_make_temp_file(image_bytes, ".png"))
    finally:
        document.close()
    return image_paths


def _tensor_list_to_serializable(
    tensors: List[torch.Tensor],
    prefix: str,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, tensor in enumerate(tensors):
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        payload.append(
            {
                "data": base64.b64encode(array.tobytes()).decode("ascii"),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "description": f"{prefix}_{idx}",
            }
        )
    return payload


class OCRModelManager:
    def __init__(
        self,
        model_id: str,
        prompt: str,
        base_size: int,
        image_size: int,
        crop_mode: bool,
        idle_timeout: int,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.idle_timeout = max(idle_timeout, 0)
        self.min_crops = max(1, MIN_CROPS)
        self.max_crops = max(self.min_crops, MAX_CROPS)
        self._patch_size = 16
        self._downsample_ratio = 4
        self._image_mean = (0.5, 0.5, 0.5)
        self._image_std = (0.5, 0.5, 0.5)
        self._pad_color = tuple(int(x * 255) for x in self._image_mean)
        self._image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self._image_mean, self._image_std),
            ]
        )

        self._model: Optional[AutoModel] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._load_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None
        self._last_used = time.monotonic()
        self._warmup_task: Optional[asyncio.Task] = None

    async def ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            self.touch()
            return

        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                self.touch()
                return

            if not torch.cuda.is_available():
                raise RuntimeError("DeepSeek-OCR requires a CUDA-enabled GPU.")

            def _load_sync():
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    use_safetensors=True,
                    _attn_implementation="flash_attention_2",
                )
                model = model.eval()
                model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
                return model, tokenizer

            model, tokenizer = await run_in_threadpool(_load_sync)

            self._model = model
            self._tokenizer = tokenizer
            self.touch()

    @staticmethod
    def _find_closest_aspect_ratio(
        aspect_ratio: float,
        target_ratios: List[Tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> Tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int,
        max_num: int,
        image_size: int,
        use_thumbnail: bool = False,
    ) -> Tuple[List[Image.Image], Tuple[int, int]]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        sorted_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, sorted_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images: List[Image.Image] = []
        for index in range(blocks):
            box = (
                (index % (target_width // image_size)) * image_size,
                (index // (target_width // image_size)) * image_size,
                ((index % (target_width // image_size)) + 1) * image_size,
                ((index // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and blocks != 1:
            processed_images.append(image.resize((image_size, image_size)))

        return processed_images, target_aspect_ratio

    def _prepare_single_image(self, image: Image.Image) -> Dict[str, Any]:
        image = image.convert("RGB")
        images_list: List[torch.Tensor] = []
        images_crop_list: List[torch.Tensor] = []
        images_spatial_crop: List[List[int]] = []

        if self.crop_mode:
            if image.size[0] <= self.image_size and image.size[1] <= self.image_size:
                crop_ratio = (1, 1)
                images_crop_raw: List[Image.Image] = []
            else:
                images_crop_raw, crop_ratio = self._dynamic_preprocess(
                    image=image,
                    min_num=self.min_crops,
                    max_num=self.max_crops,
                    image_size=self.image_size,
                )
        else:
            crop_ratio = (1, 1)
            images_crop_raw = []

        global_view = ImageOps.pad(
            image,
            (self.base_size, self.base_size),
            color=self._pad_color,
        )
        images_list.append(self._image_transform(global_view).to(torch.bfloat16))
        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])

        if images_crop_raw:
            for patch in images_crop_raw:
                images_crop_list.append(
                    self._image_transform(patch).to(torch.bfloat16)
                )

        if images_list:
            images_ori = torch.stack(images_list, dim=0)
        else:
            images_ori = torch.zeros(
                (1, 3, self.base_size, self.base_size), dtype=torch.bfloat16
            )

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros(
                (1, 3, self.image_size, self.image_size), dtype=torch.bfloat16
            )

        return {
            "patches": images_crop,
            "global": images_ori,
            "crop_shape": crop_ratio,
            "original_size": image.size,
        }

    def _extract_image_embeddings_sync(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        assert self._model is not None
        base_model = self._model.model  # type: ignore[attr-defined]
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        with Image.open(image_path) as pil_image:
            prepared = self._prepare_single_image(pil_image)
        crop_width, crop_height = prepared["crop_shape"]

        patches = prepared["patches"].to(device=device, dtype=dtype)
        image_ori = prepared["global"].to(device=device, dtype=dtype)

        sam_model = base_model.sam_model
        vision_model = base_model.vision_model
        projector = base_model.projector
        image_newline = base_model.image_newline.to(device=device, dtype=dtype)
        view_separator = base_model.view_seperator.to(device=device, dtype=dtype)

        embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            if torch.sum(patches).item() != 0:
                local_features_1 = sam_model(patches)
                local_features_2 = vision_model(patches, local_features_1)
                local_features = torch.cat(
                    (
                        local_features_2[:, 1:],
                        local_features_1.flatten(2).permute(0, 2, 1),
                    ),
                    dim=-1,
                )
                local_features = projector(local_features)

                global_features_1 = sam_model(image_ori)
                global_features_2 = vision_model(image_ori, global_features_1)
                global_features = torch.cat(
                    (
                        global_features_2[:, 1:],
                        global_features_1.flatten(2).permute(0, 2, 1),
                    ),
                    dim=-1,
                )
                global_features = projector(global_features)

                _, hw, n_dim = global_features.shape
                h = w = int(math.sqrt(hw))
                global_features = (
                    global_features.view(-1, h, w, n_dim)
                    .squeeze(0)
                    .contiguous()
                )
                global_features = torch.cat(
                    [
                        global_features,
                        image_newline[None, None, :].expand(h, 1, n_dim),
                    ],
                    dim=1,
                ).view(-1, n_dim)

                _, hw2, n_dim2 = local_features.shape
                h2 = w2 = int(math.sqrt(hw2))
                local_features = (
                    local_features.view(
                        crop_height, crop_width, h2, w2, n_dim2
                    )
                    .permute(0, 2, 1, 3, 4)
                    .reshape(crop_height * h2, crop_width * w2, n_dim2)
                )
                local_features = torch.cat(
                    [
                        local_features,
                        image_newline[None, None, :].expand(
                            crop_height * h2, 1, n_dim2
                        ),
                    ],
                    dim=1,
                ).view(-1, n_dim2)

                tokens = torch.cat(
                    [local_features, global_features, view_separator[None, :]], dim=0
                )
                embeddings.append(tokens.to(torch.float32).cpu())
            else:
                global_features_1 = sam_model(image_ori)
                global_features_2 = vision_model(image_ori, global_features_1)
                global_features = torch.cat(
                    (
                        global_features_2[:, 1:],
                        global_features_1.flatten(2).permute(0, 2, 1),
                    ),
                    dim=-1,
                )
                global_features = projector(global_features)

                _, hw, n_dim = global_features.shape
                h = w = int(math.sqrt(hw))
                global_features = (
                    global_features.view(-1, h, w, n_dim)
                    .squeeze(0)
                    .contiguous()
                )
                global_features = torch.cat(
                    [
                        global_features,
                        image_newline[None, None, :].expand(h, 1, n_dim),
                    ],
                    dim=1,
                ).view(-1, n_dim)

                tokens = torch.cat(
                    [global_features, view_separator[None, :]], dim=0
                )
                embeddings.append(tokens.to(torch.float32).cpu())

        return {
            "embeddings": embeddings,
            "token_count": sum(tensor.shape[0] for tensor in embeddings),
            "crop_shape": (crop_width, crop_height),
            "original_size": prepared["original_size"],
        }

    def _infer_sync(
        self,
        image_path: str,
        prompt: Optional[str],
    ) -> str:
        assert self._model is not None
        assert self._tokenizer is not None

        output_dir = Path(tempfile.mkdtemp(prefix="deepseek-ocr-out-"))
        try:
            result = self._model.infer(
                self._tokenizer,
                prompt=prompt or self.prompt,
                image_file=image_path,
                output_path=str(output_dir),
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                save_results=False,
                test_compress=False,
                eval_mode=True,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        return result.strip()

    async def unload(self, force: bool = False) -> bool:
        async with self._load_lock:
            if self._model is None:
                return False
            elapsed = time.monotonic() - self._last_used
            if not force and self.idle_timeout and elapsed < self.idle_timeout:
                return False
            model = self._model
            tokenizer = self._tokenizer
            self._model = None
            self._tokenizer = None

        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

        del model
        del tokenizer
        with suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return True

    def touch(self) -> None:
        self._last_used = time.monotonic()
        if self.idle_timeout:
            if self._idle_task and not self._idle_task.done():
                self._idle_task.cancel()
            loop = asyncio.get_running_loop()
            self._idle_task = loop.create_task(self._idle_watchdog())

    async def _idle_watchdog(self) -> None:
        try:
            await asyncio.sleep(self.idle_timeout)
            if time.monotonic() - self._last_used >= self.idle_timeout:
                await self.unload(force=True)
        except asyncio.CancelledError:
            return

    async def infer_on_image(self, image_path: str, prompt: Optional[str]) -> str:
        await self.ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        async with self._infer_lock:
            result = await run_in_threadpool(
                lambda: self._infer_sync(image_path, prompt)
            )

        self.touch()
        return result

    async def embeddings_on_image(
        self,
        image_path: str,
        prompt: Optional[str],
        include_text: bool,
    ) -> Dict[str, Any]:
        await self.ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        async with self._infer_lock:
            embeddings_info, text_result = await run_in_threadpool(
                lambda: (
                    self._extract_image_embeddings_sync(image_path),
                    self._infer_sync(image_path, prompt) if include_text else None,
                )
            )

        if include_text and text_result is not None:
            embeddings_info["text"] = text_result

        self.touch()
        return embeddings_info

    async def run_image_bytes(self, data: bytes, suffix: str, prompt: Optional[str]) -> str:
        temp_path = _make_temp_file(data, suffix)
        try:
            return await self.infer_on_image(temp_path, prompt)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)

    async def run_image_embeddings(
        self,
        data: bytes,
        suffix: str,
        prompt: Optional[str],
        include_text: bool,
    ) -> Dict[str, Any]:
        temp_path = _make_temp_file(data, suffix)
        try:
            return await self.embeddings_on_image(temp_path, prompt, include_text)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)

    async def run_pdf_bytes(self, data: bytes, prompt: Optional[str]) -> List[Dict[str, str]]:
        image_paths = _pdf_bytes_to_image_paths(data, PDF_DPI)
        if MAX_PDF_PAGES and len(image_paths) > MAX_PDF_PAGES:
            for extra_path in image_paths[MAX_PDF_PAGES:]:
                with suppress(FileNotFoundError):
                    os.unlink(extra_path)
            image_paths = image_paths[:MAX_PDF_PAGES]

        results: List[Dict[str, str]] = []
        try:
            for page_index, image_path in enumerate(image_paths):
                text = await self.infer_on_image(image_path, prompt)
                results.append({"page": page_index, "text": text})
        finally:
            for path in image_paths:
                with suppress(FileNotFoundError):
                    os.unlink(path)
        return results

    async def run_pdf_embeddings(
        self,
        data: bytes,
        prompt: Optional[str],
        include_text: bool,
    ) -> List[Dict[str, Any]]:
        image_paths = _pdf_bytes_to_image_paths(data, PDF_DPI)
        if MAX_PDF_PAGES and len(image_paths) > MAX_PDF_PAGES:
            for extra_path in image_paths[MAX_PDF_PAGES:]:
                with suppress(FileNotFoundError):
                    os.unlink(extra_path)
            image_paths = image_paths[:MAX_PDF_PAGES]

        results: List[Dict[str, Any]] = []
        try:
            for page_index, image_path in enumerate(image_paths):
                info = await self.embeddings_on_image(image_path, prompt, include_text)
                info["page"] = page_index
                results.append(info)
        finally:
            for path in image_paths:
                with suppress(FileNotFoundError):
                    os.unlink(path)
        return results

    def start_background_warmup(self) -> None:
        if self._warmup_task and not self._warmup_task.done():
            return
        loop = asyncio.get_running_loop()
        self._warmup_task = loop.create_task(self.ensure_loaded())


model_manager = OCRModelManager(
    model_id=MODEL_ID,
    prompt=DEFAULT_PROMPT,
    base_size=BASE_SIZE,
    image_size=IMAGE_SIZE,
    crop_mode=CROP_MODE,
    idle_timeout=IDLE_TIMEOUT_SECONDS,
)

app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    if os.getenv("DEEPSEEK_OCR_PRELOAD", "1") not in {"0", "false", "no"}:
        model_manager.start_background_warmup()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await model_manager.unload(force=True)


@app.get("/health")
def health() -> PlainTextResponse:
    return PlainTextResponse("ok", status_code=200)


@app.post("/warmup")
async def warmup() -> JSONResponse:
    await model_manager.ensure_loaded()
    return JSONResponse({"status": "ok", "detail": "model ready"})


@app.post("/idle")
async def force_idle() -> JSONResponse:
    unloaded = await model_manager.unload(force=True)
    return JSONResponse({"status": "ok", "detail": "model unloaded" if unloaded else "model already idle"})


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if not suffix:
        content_type = (file.content_type or "").lower()
        suffix = CONTENT_TYPE_TO_SUFFIX.get(content_type, "")

    try:
        if suffix == ".pdf":
            pages = await model_manager.run_pdf_bytes(data, prompt)
            return JSONResponse(
                {
                    "status": "ok",
                    "type": "pdf",
                    "page_count": len(pages),
                    "pages": pages,
                }
            )

        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            text = await model_manager.run_image_bytes(data, suffix or ".png", prompt)
            return JSONResponse(
                {
                    "status": "ok",
                    "type": "image",
                    "text": text,
                }
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{suffix or 'unknown'}'. Upload a PDF or supported image.",
    )


@app.post("/ocr/embeddings")
async def ocr_embeddings(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    include_text: bool = False,
) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if not suffix:
        content_type = (file.content_type or "").lower()
        suffix = CONTENT_TYPE_TO_SUFFIX.get(content_type, "")

    try:
        if suffix == ".pdf":
            pages = await model_manager.run_pdf_embeddings(data, prompt, include_text)
            response_pages: List[Dict[str, Any]] = []
            total_tokens = 0

            for page_info in pages:
                embeddings = page_info["embeddings"]
                page_index = page_info["page"]
                serialized = _tensor_list_to_serializable(
                    embeddings, f"vision_tokens_page_{page_index}"
                )
                del page_info["embeddings"]
                total_tokens += page_info["token_count"]

                page_payload: Dict[str, Any] = {
                    "page": page_index,
                    "token_count": page_info["token_count"],
                    "embeddings": serialized,
                    "metadata": {
                        "crop_shape": list(page_info["crop_shape"]),
                        "original_size": list(page_info["original_size"]),
                    },
                }
                if include_text and "text" in page_info:
                    page_payload["text"] = page_info["text"]
                response_pages.append(page_payload)

            return JSONResponse(
                {
                    "status": "ok",
                    "type": "pdf",
                    "page_count": len(response_pages),
                    "total_tokens": total_tokens,
                    "pages": response_pages,
                }
            )

        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            info = await model_manager.run_image_embeddings(
                data, suffix or ".png", prompt, include_text
            )
            serialized = _tensor_list_to_serializable(
                info["embeddings"], "vision_tokens_image_0"
            )
            del info["embeddings"]
            payload: Dict[str, Any] = {
                "status": "ok",
                "type": "image",
                "token_count": info["token_count"],
                "embeddings": serialized,
                "metadata": {
                    "crop_shape": list(info["crop_shape"]),
                    "original_size": list(info["original_size"]),
                },
            }
            if include_text and "text" in info:
                payload["text"] = info["text"]
            return JSONResponse(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{suffix or 'unknown'}'. Upload a PDF or supported image.",
    )
