"""
Slot mapping cho outfit pairing — derived from product_type_name của H&M dataset (113 unique values).

Slot taxonomy:
- top: phần thân trên, lớp trong cùng / chính (shirt, t-shirt, blouse, hoodie, sweater, polo, top, vest top, cardigan, bodysuit, tailored waistcoat)
- outerwear: lớp ngoài (blazer, coat, jacket, outdoor waistcoat)
- bottom: phần thân dưới (trousers, skirt, shorts, leggings, dungarees, outdoor trousers)
- dress: full-body single piece (dress, jumpsuit/playsuit, garment set) — KHÔNG pair với top/bottom
- shoe: footwear
- accessory: phụ kiện (bag, belt, hat, scarf, jewelry, watch, glasses, gloves...)
- inner: underwear / inner layer (bra, underwear bottom/body/corset, socks, long john, underdress)
- nightwear: pyjama, robe, night gown — KHÔNG dùng cho outfit pairing (sẽ bị rule_0 loại)
- swim: bikini, swimsuit, sarong
- other: items không phải fashion (giftbox, marker pen, dog wear, sewing kit, furniture...)

Pairing logic (allowed cross-slot combinations):
- top + (bottom | outerwear | shoe | accessory)
- bottom + (outerwear | shoe | accessory)
- dress + (outerwear | shoe | accessory) — KHÔNG đi với top/bottom vì dress đã cover full body
- outerwear + (shoe | accessory)
- shoe + accessory
- swim + (shoe | accessory)
- accessory + accessory (CHO PHÉP cross-PT trong cùng outfit, vd. necklace + belt + watch; rule_2_strict đảm bảo không same-PT)

Forbidden pairs (same slot OR semantic conflict):
- (top, top), (bottom, bottom), (outerwear, outerwear), (shoe, shoe), (dress, dress), (inner, inner), (swim, swim)
- (dress, top), (dress, bottom): dress là full body
- (swim, top), (swim, bottom), (swim, outerwear): swimwear context
- bất kỳ slot kết hợp với inner / nightwear / other → drop
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
    "Garment Set": "dress",

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

    # swim — context riêng (chỉ pair với shoe/accessory)
    "Bikini top": "swim",
    "Sarong": "swim",
    "Swimsuit": "swim",
    "Swimwear bottom": "swim",
    "Swimwear set": "swim",

    # other — không phải fashion item, không vào graph
    "Clothing mist": "other",
    "Costumes": "other",
    "Dog Wear": "other",
    "Dog wear": "other",
    "Giftbox": "other",
    "Marker pen": "other",
    "Sewing kit": "other",
    "Side table": "other",
    "Stain remover spray": "other",
    "Umbrella": "other",
    "Unknown": "other",
    "Washing bag": "other",
    "Waterbottle": "other",
    "Wood balls": "other",
    "Zipper head": "other",
    "Furniture": "other",
}


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
    "Unknown": "other",
}


def get_slot(product_type: str, product_group: str = "") -> str:
    pt_slot = PT_TO_SLOT.get(product_type)
    if pt_slot is not None and pt_slot != "other":
        return pt_slot
    if product_group:
        pg_slot = PG_TO_SLOT.get(product_group)
        if pg_slot is not None:
            return pg_slot
    return pt_slot if pt_slot is not None else "other"


def slot_pair_allowed(slot_a: str, slot_b: str) -> bool:
    return frozenset([slot_a, slot_b]) in ALLOWED_SLOT_PAIRS
