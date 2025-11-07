import os
import glob
from typing import Optional, Sequence, Any, List, Dict
from PIL import Image, ImageOps

_VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")


def _first_pathlike(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("path", "image_path", "img_path", "filepath", "file"):
            v = x.get(k)
            if isinstance(v, str) and v:
                return v
    if isinstance(x, (list, tuple)):
        for e in x:
            if isinstance(e, str) and e.lower().endswith(_VALID_EXTS):
                return e
        if len(x) > 0:
            return str(x[0])
        return ""
    return str(x) if x is not None else ""


def resolve_image_path(image_id: Any, base_dir: Optional[Any] = None) -> str:
    img = _first_pathlike(image_id)
    bd = _first_pathlike(base_dir) if base_dir is not None else ""
    if os.path.isabs(img):
        return img
    if bd and os.path.isdir(bd):
        return os.path.join(bd, img)
    return img


class ImageResolver:
    """
    Robustly resolves image paths given (image_id, base_dir) and a list of root directories.
    Caches results to be fast across many lookups.
    """
    def __init__(self, roots: Optional[Sequence[str]] = None, extra_try: Optional[Sequence[str]] = None):
        roots = roots or []
        self.roots: List[str] = [r for r in [*(roots or []), *(extra_try or [])] if r and os.path.isdir(r)]
        self.cache: Dict[str, Optional[str]] = {}

    def _exists(self, p: str) -> bool:
        return p and os.path.isfile(p)

    def _try_roots_join(self, rel: str) -> Optional[str]:
        for r in self.roots:
            cand = os.path.join(r, rel)
            if self._exists(cand):
                return cand
        return None

    def _search_basename(self, name: str) -> Optional[str]:
        # recursive glob for the basename in each root
        for r in self.roots:
            hits = glob.glob(os.path.join(r, "**", name), recursive=True)
            if hits:
                return hits[0]
        return None

    def resolve(self, image_id: Any, base_dir: Optional[Any] = None, pkl_dir: Optional[str] = None) -> Optional[str]:
        key = f"{image_id}|{base_dir}"
        if key in self.cache:
            return self.cache[key]

        # 1) direct resolution using base_dir
        p = resolve_image_path(image_id, base_dir)
        if self._exists(p):
            self.cache[key] = p
            return p

        # 2) if relative, try join with each root
        if not os.path.isabs(p):
            hit = self._try_roots_join(p)
            if hit:
                self.cache[key] = hit
                return hit

        # 3) try PKL directory
        if pkl_dir and not os.path.isabs(p):
            cand = os.path.join(pkl_dir, p)
            if self._exists(cand):
                self.cache[key] = cand
                return cand

        # 4) try CWD
        if not os.path.isabs(p) and self._exists(p):
            self.cache[key] = p
            return p

        # 5) basename search
        name = os.path.basename(p)
        hit = self._search_basename(name)
        self.cache[key] = hit  # may be None
        return hit


def load_padchest_image(image_path_or_id: Any, base_dir: Optional[Any] = None) -> Image.Image:
    """
    Loads an image path (already resolved or not) and returns a PIL RGB image.
    Accepts strings, tuples, lists, dicts.
    """
    p = image_path_or_id
    if not isinstance(p, str) or not os.path.isfile(p):
        p = resolve_image_path(image_path_or_id, base_dir)
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(p)
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
