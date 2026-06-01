"""
Slot mapping cho outfit pairing — 3-tier fallback derived from H&M dataset.

Slot taxonomy:
- top: phần thân trên, lớp trong cùng / chính (shirt, t-shirt, blouse, hoodie, sweater, polo, top, vest top, cardigan, bodysuit, tailored waistcoat)
- outerwear: lớp ngoài (blazer, coat, jacket, outdoor waistcoat)
- bottom: phần thân dưới (trousers, skirt, shorts, leggings, dungarees, outdoor trousers)
- dress: full-body single piece (dress, jumpsuit/playsuit) — KHÔNG pair với top/bottom
- shoe: footwear
- accessory: phụ kiện (bag, belt, hat, scarf, jewelry, watch, glasses, gloves...)
- inner: underwear / inner layer (bra, underwear bottom/body/corset, socks, long john, underdress)
- nightwear: pyjama, robe, night gown, garment set (đa số là pyjama set) — KHÔNG dùng cho outfit pairing
- swim: bikini, swimsuit
- other: items không phải fashion (giftbox, marker pen, dog wear, sewing kit, furniture, costumes, sarong=noisy)

3-tier fallback in get_slot():
- Tier 1: PT_TO_SLOT[product_type_name] — explicit (113 PTs mapped, "Unknown" KHÔNG có trong dict để cho phép fallback)
- Tier 2: PG_TO_SLOT[product_group_name] — chỉ kích hoạt khi PT không trong dict
- Tier 3: GG_TO_SLOT[garment_group_name] + prod_name hint cho mixed GG — final fallback cho 111 Unknown PT items
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Set


PT_TO_SLOT: Dict[str, str] = {
    # top — lớp trong / chính
    "Blouse": "top",
    "Bodysuit": "top",
    "Cardigan": "top",
    "Hoodie": "top",
    "Polo shirt": "top",
    "Shirt": "top",
    "Sweater": "top",
    "T-shirt": "top",
    "Top": "top",
    "Vest top": "top",
    "Tailored Waistcoat": "top",

    # outerwear — lớp ngoài
    "Blazer": "outerwear",
    "Coat": "outerwear",
    "Jacket": "outerwear",
    "Outdoor Waistcoat": "outerwear",

    # bottom — phần thân dưới
    "Dungarees": "bottom",
    "Leggings/Tights": "bottom",
    "Outdoor trousers": "bottom",
    "Shorts": "bottom",
    "Skirt": "bottom",
    "Trousers": "bottom",

    # dress — full body, không tách top+bottom
    "Dress": "dress",
    "Jumpsuit/Playsuit": "dress",

    # shoe — footwear
    "Ballerinas": "shoe",
    "Bootie": "shoe",
    "Boots": "shoe",
    "Flat shoe": "shoe",
    "Flat shoes": "shoe",
    "Flip flop": "shoe",
    "Heeled sandals": "shoe",
    "Heels": "shoe",
    "Other shoe": "shoe",
    "Pumps": "shoe",
    "Sandals": "shoe",
    "Slippers": "shoe",
    "Sneakers": "shoe",
    "Wedge": "shoe",

    # accessory — phụ kiện (cross-PT cho phép)
    "Alice band": "accessory",
    "Backpack": "accessory",
    "Bag": "accessory",
    "Beanie": "accessory",
    "Belt": "accessory",
    "Bracelet": "accessory",
    "Braces": "accessory",
    "Bucket hat": "accessory",
    "Cap": "accessory",
    "Cap/peaked": "accessory",
    "Cross-body bag": "accessory",
    "Earring": "accessory",
    "Earrings": "accessory",
    "Eyeglasses": "accessory",
    "Felt hat": "accessory",
    "Gloves": "accessory",
    "Hair clip": "accessory",
    "Hair string": "accessory",
    "Hair ties": "accessory",
    "Hair/alice band": "accessory",
    "Hairband": "accessory",
    "Hat/beanie": "accessory",
    "Hat/brim": "accessory",
    "Headband": "accessory",
    "Keychain": "accessory",
    "Mobile case": "accessory",
    "Necklace": "accessory",
    "Other accessories": "accessory",
    "Ring": "accessory",
    "Scarf": "accessory",
    "Shoulder bag": "accessory",
    "Straw hat": "accessory",
    "Sunglasses": "accessory",
    "Tie": "accessory",
    "Tote bag": "accessory",
    "Wallet": "accessory",
    "Watch": "accessory",
    "Weekend/Gym bag": "accessory",
    "Wireless earphone case": "accessory",

    # inner — underwear / inner layer (loại khỏi outfit pairing)
    "Bra": "inner",
    "Bra extender": "inner",
    "Nipple covers": "inner",
    "Socks": "inner",
    "Underdress": "inner",
    "Underwear Tights": "inner",
    "Underwear body": "inner",
    "Underwear bottom": "inner",
    "Underwear corset": "inner",
    "Underwear set": "inner",
    "Long John": "inner",

    # nightwear — loại
    "Night gown": "nightwear",
    "Pyjama bottom": "nightwear",
    "Pyjama jumpsuit/playsuit": "nightwear",
    "Pyjama set": "nightwear",
    "Robe": "nightwear",
    "Garment Set": "nightwear",  # data inspection 46 items: toàn pyjama/athletic set, không phải dress full-body

    # swim — context riêng (chỉ pair với shoe/accessory)
    "Bikini top": "swim",
    "Swimsuit": "swim",
    "Swimwear bottom": "swim",
    "Swimwear set": "swim",

    # other — không phải fashion item / data noise, không vào graph
    "Clothing mist": "other",
    "Costumes": "other",  # 4 items label sai (t-shirt + blazer), data noise
    "Dog Wear": "other",
    "Dog wear": "other",
    "Giftbox": "other",
    "Marker pen": "other",
    "Sarong": "other",  # 63 items mixed (wrap + kaftan + scarf + hat + bag) → drop để tránh noise
    "Sewing kit": "other",
    "Side table": "other",
    "Stain remover spray": "other",
    "Umbrella": "other",
    "Washing bag": "other",
    "Waterbottle": "other",
    "Wood balls": "other",
    "Zipper head": "other",
    "Furniture": "other",
    # NOTE: "Unknown" cố tình KHÔNG có trong dict để get_slot() chuyển sang Tier 2/3 fallback
}


PG_TO_SLOT: Dict[str, str] = {
    "Garment Upper body": "top",
    "Garment Lower body": "bottom",
    "Garment Full body": "dress",
    "Shoes": "shoe",
    "Accessories": "accessory",
    "Bags": "accessory",
    "Underwear": "inner",
    "Socks & Tights": "inner",
    "Swimwear": "swim",
    "Nightwear": "nightwear",
    "Furniture": "other",
    "Items": "other",
    "Stationery": "other",
    "Garment and Shoe care": "other",
    # "Unknown" KHÔNG mapped → fall through Tier 3
}


GG_TO_SLOT: Dict[str, str] = {
    # Tier 3 fallback, derived empirically từ inspect_unknown_pt.py trên 111 Unknown PT items.
    # Verified clean (single-slot): ~100/111 items đúng.
    "Accessories": "accessory",
    "Blouses": "top",
    "Dressed": "outerwear",  # samples = Womens Tailoring suit jackets
    "Dresses Ladies": "dress",
    "Knitwear": "top",
    "Outdoor": "outerwear",
    "Shirts": "top",
    "Shoes": "shoe",
    "Shorts": "bottom",
    "Skirts": "bottom",
    "Socks and Tights": "inner",
    "Swimwear": "swim",
    "Trousers": "bottom",
    "Trousers Denim": "bottom",
    "Under-, Nightwear": "inner",  # 39 Unknown PT items đều ở Womens Lingerie sections → inner
    "Special Offers": "other",
    # NOTE: "Jersey Basic" + "Jersey Fancy" + "Unknown" KHÔNG mapped — mixed slot, dùng prod_name hint
}


MIXED_GG_NEEDS_PROD_NAME: Set[str] = {"Jersey Basic", "Jersey Fancy", "Unknown"}

INNER_SECTIONS: Set[str] = {
    "Men Underwear",
    "Womens Lingerie",
    "Womens Nightwear, Socks & Tigh",
}

INNER_DEPARTMENTS: Set[str] = {
    "Ladies Sport Bras",
    "Casual Lingerie",
    "Clean Lingerie",
    "Expressive Lingerie",
    "Functional Lingerie",
    "Underwear Jersey",
    "Underwear Woven",
    "Tights basic",
    "Socks Bin",
    "Socks Wall",
    "Mama Lingerie",
    "Shopbasket Lingerie",
    "Shopbasket Socks",
    "Nursing",
    "UW",
}


def _section_dept_hint(section_name: str, department_name: str) -> str | None:
    if section_name in INNER_SECTIONS or department_name in INNER_DEPARTMENTS:
        return "inner"
    return None


PROD_NAME_BOTTOM_KEYWORDS = ("trouser", " trs", "tights", "legging", "shorts", "hotpant", "skirt", "jeans", "denim")
PROD_NAME_DRESS_KEYWORDS = ("dress",)
PROD_NAME_OUTERWEAR_KEYWORDS = ("jacket", "blazer", "coat", "parka", "puffer", "padded")
PROD_NAME_TOP_KEYWORDS = ("tee", "tank", "hood", "shirt", " top", "sweater", "jumper", "polo", "cardigan", "bodysuit", "tee(", "stringer")
# NOTE: bỏ "r-neck", "v-neck", "crew", "henley" — quá generic, match cả tee AND undershirt (R-NECK SS BASIC).
# Để section/dept hint quyết định context cho các tên không có keyword rõ ràng.


def _prod_name_hint(prod_name: str) -> str | None:
    if not prod_name:
        return None
    p = prod_name.lower().strip()
    if any(k in p for k in (" bra ", "bralette")) or p.endswith(" bra") or p.startswith("bra "):
        return "inner"
    if any(k in p for k in PROD_NAME_BOTTOM_KEYWORDS):
        return "bottom"
    if any(k in p for k in PROD_NAME_DRESS_KEYWORDS):
        return "dress"
    if any(k in p for k in PROD_NAME_OUTERWEAR_KEYWORDS):
        return "outerwear"
    if any(k in p for k in PROD_NAME_TOP_KEYWORDS):
        return "top"
    return None


PT_FAMILY: Dict[str, str] = {
    # Headwear family — H&M tách Cap vs Cap/peaked vs Hat/* nhưng đeo cùng 1 cái thôi
    "Cap": "headwear",
    "Cap/peaked": "headwear",
    "Bucket hat": "headwear",
    "Felt hat": "headwear",
    "Hat/beanie": "headwear",
    "Hat/brim": "headwear",
    "Straw hat": "headwear",
    "Beanie": "headwear",

    # Hair accessory family — Alice band / Headband / Hairband / Hair/alice band đều là band tóc
    "Hair clip": "hair_acc",
    "Hair string": "hair_acc",
    "Hair ties": "hair_acc",
    "Hair/alice band": "hair_acc",
    "Hairband": "hair_acc",
    "Headband": "hair_acc",
    "Alice band": "hair_acc",

    # Earring family — Earring (single) vs Earrings (set) cùng nghĩa
    "Earring": "earring",
    "Earrings": "earring",

    # Bag family — Bag / Backpack / Cross-body / Shoulder / Tote / Weekend đều là túi
    "Bag": "bag",
    "Backpack": "bag",
    "Cross-body bag": "bag",
    "Shoulder bag": "bag",
    "Tote bag": "bag",
    "Weekend/Gym bag": "bag",

    # Shoe singular/plural duplicates (đã được slot=shoe + same_pt cover phần lớn, thêm cho an toàn)
    "Flat shoe": "flat_shoe",
    "Flat shoes": "flat_shoe",

    # Dog wear case-sensitivity
    "Dog wear": "dog_wear",
    "Dog Wear": "dog_wear",
}


def same_pt_family(pt_a: str, pt_b: str) -> bool:
    if not pt_a or not pt_b:
        return False
    fam_a = PT_FAMILY.get(pt_a)
    fam_b = PT_FAMILY.get(pt_b)
    return fam_a is not None and fam_a == fam_b


ALLOWED_SLOT_PAIRS: Set[FrozenSet[str]] = {
    frozenset(["top", "bottom"]),
    frozenset(["top", "outerwear"]),
    frozenset(["top", "shoe"]),
    frozenset(["top", "accessory"]),
    frozenset(["bottom", "outerwear"]),
    frozenset(["bottom", "shoe"]),
    frozenset(["bottom", "accessory"]),
    frozenset(["outerwear", "shoe"]),
    frozenset(["outerwear", "accessory"]),
    frozenset(["dress", "outerwear"]),
    frozenset(["dress", "shoe"]),
    frozenset(["dress", "accessory"]),
    frozenset(["shoe", "accessory"]),
    frozenset(["swim", "shoe"]),
    frozenset(["swim", "accessory"]),
    frozenset(["accessory", "accessory"]),
}


def get_slot(
    product_type: str,
    product_group: str = "",
    garment_group: str = "",
    prod_name: str = "",
    section_name: str = "",
    department_name: str = "",
) -> str:
    """4-tier fallback cho slot lookup:
    - Tier 1: PT_TO_SLOT (trusted tuyệt đối khi có entry, kể cả "other")
    - Tier 2: PG_TO_SLOT (chỉ khi PT không trong dict, e.g. PT="Unknown")
    - Tier 3: GG_TO_SLOT (khi PG cũng Unknown). Với mixed GG (Jersey Basic/Fancy):
        - Tier 3a: section/department hint (catch Lingerie/Underwear context → inner)
        - Tier 3b: prod_name keyword hint
    - Default: "other"
    """
    pt_slot = PT_TO_SLOT.get(product_type)
    if pt_slot is not None:
        return pt_slot
    if product_group:
        pg_slot = PG_TO_SLOT.get(product_group)
        if pg_slot is not None:
            return pg_slot
    if garment_group:
        if garment_group in MIXED_GG_NEEDS_PROD_NAME:
            name_hint = _prod_name_hint(prod_name)
            if name_hint is not None:
                return name_hint
            sect_hint = _section_dept_hint(section_name, department_name)
            if sect_hint is not None:
                return sect_hint
        else:
            gg_slot = GG_TO_SLOT.get(garment_group)
            if gg_slot is not None:
                return gg_slot
    return "other"


def slot_pair_allowed(slot_a: str, slot_b: str) -> bool:
    return frozenset([slot_a, slot_b]) in ALLOWED_SLOT_PAIRS
