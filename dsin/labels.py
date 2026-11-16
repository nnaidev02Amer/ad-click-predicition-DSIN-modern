"""Human-friendly display labels for the dashboard.

Only DISPLAY text lives here; the model always receives the original encoded ids.

* Category / brand names: the raw Taobao ids are anonymous numbers and the dataset ships no names. For the
  synthetic DEMO data we attach illustrative, fictional labels (assigned by position in the sorted id list,
  so they are stable). They carry no real-world meaning. In real mode the labels fall back to "Category <id>".
* Age: the repository never defines what each age code means and the synthetic generator draws it at random,
  so no age-in-years -> code mapping exists (see age_to_model_category): the model gets a fixed default code.
* Gender: codes 1 / 2 are shown as Male / Female (the public Taobao dataset description; not verifiable
  inside this repository - the synthetic generator just draws 1 or 2).
"""
from __future__ import annotations

CATEGORY_NAMES = [
    'Sarees', 'Kurtas', 'Smartphones', 'Earphones', 'Laptops', 'Mixer Grinders', 'Pressure Cookers', 'Rice',
    'Cooking Oil', 'Cricket Bats', 'Sports Shoes', 'Sandals', 'Bags', 'Televisions', 'Refrigerators', 'Spices',
    'Tea & Coffee', 'Bedsheets', 'Wrist Watches', 'Backpacks', 'Gold-plated Jewellery', 'Kids Clothing', 'Books',
    'Water Bottles', 'Air Coolers', 'Mobile Covers', 'Lehengas', 'Yoga Mats', 'Dry Fruits', 'Table Fans']
# fictional Indian-style brand names (NOT real companies) so that no real brand is tied to synthetic records
_PREFIX = ['Kaveri', 'Ganga', 'Surya', 'Chandra', 'Lakshmi', 'Bharat', 'Desi', 'Tulsi', 'Neelam', 'Rangoli', 'Mysore', 'Himalaya']
_SUFFIX = ['Creations', 'Mart', 'Works', 'Select', 'Home']
BRAND_NAMES = [f'{p} {s}' for s in _SUFFIX for p in _PREFIX]          # 60 fictional brands

GENDER_LABELS = {1: 'Male', 2: 'Female'}


AGE_MIN, AGE_MAX = 13, 100


def age_label(code, synthetic: bool) -> str:
    return f'Age category {int(code)}' + (' (demo)' if synthetic else '')


def age_to_model_category(age_years: int, choices):
    """Return (model age_level code, mapping_established).

    No mapping from age in years to the dataset's age_level codes can be established here: the synthetic
    generator draws age_level uniformly at random from 1..6 (dsin/synthetic.py), the observed click rate is
    unrelated to it, and the repository nowhere defines the codes' meaning. The model's age input therefore
    uses a FIXED default code (the first valid one) for every age, and the UI says so. The age typed by the
    user is recorded for display only.
    """
    if not AGE_MIN <= int(age_years) <= AGE_MAX:
        raise ValueError(f'Age must be between {AGE_MIN} and {AGE_MAX}.')
    return choices[0], False


def gender_label(code) -> str:
    return GENDER_LABELS.get(int(code), f'Gender code {int(code)}')


class Labels:
    """Two-way map between raw category/brand ids and display names."""

    def __init__(self, cate_classes, brand_classes, synthetic: bool):
        self.synthetic = synthetic
        self.cate = self._build(sorted({int(c) for c in cate_classes}), CATEGORY_NAMES, 'Category', synthetic)
        self.brand = self._build([b for b in sorted({int(b) for b in brand_classes}) if b >= 0], BRAND_NAMES, 'Brand', synthetic)
        self.brand[-1] = 'No brand'
        self.cate_rev = {v: k for k, v in self.cate.items()}
        self.brand_rev = {v: k for k, v in self.brand.items()}

    @staticmethod
    def _build(ids, names, prefix, synthetic):
        out = {}
        for i, raw in enumerate(ids):
            if synthetic:
                name = names[i % len(names)]
                out[raw] = name if i < len(names) else f'{name} {i // len(names) + 1}'
            else:
                out[raw] = f'{prefix} {raw}'
        return out

    def cate_name(self, raw) -> str:
        return self.cate.get(int(raw), f'Category {int(raw)}')

    def brand_name(self, raw) -> str:
        return self.brand.get(int(raw), f'Brand {int(raw)}')

    def pair_label(self, cate_raw, brand_raw) -> str:
        return f'{self.cate_name(cate_raw)} · {self.brand_name(brand_raw)}'
