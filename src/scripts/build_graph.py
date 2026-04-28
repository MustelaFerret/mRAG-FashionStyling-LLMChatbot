import pandas as pd
from collections import Counter
import itertools
import gc
import time

import os

META_FILE = os.getenv("META_FILE", "data/processed/dataset_final_qwen_filled.csv")
TRANS_FILE = os.getenv("TRANS_FILE", "data/raw/transactions_train.csv")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/processed/final_outfit_graph.csv")

def main():
    print("graph build started")
    start_time = time.time()

    print("loading metadata")
    meta_cols = [
        'article_id', 'product_type_name', 'product_group_name', 'garment_group_name',
        'department_name', 'index_name', 'section_name', 'colour_group_name', 
        'fit', 'occasion', 'seasonality'
    ]
    df_meta = pd.read_csv(META_FILE, usecols=meta_cols, dtype={'article_id': str})
    df_meta['article_id'] = df_meta['article_id'].str.zfill(10)
    valid_ids = set(df_meta['article_id'])

    print("loading transactions")
    df_trans = pd.read_csv(TRANS_FILE, usecols=["t_dat", "customer_id", "article_id"], dtype={"article_id": str})
    df_trans["article_id"] = df_trans["article_id"].str.zfill(10)
    df_trans = df_trans[df_trans["article_id"].isin(valid_ids)]
    df_trans = df_trans.drop_duplicates(subset=['customer_id', 't_dat', 'article_id'])

    print("building baskets")
    baskets = df_trans.groupby(['customer_id', 't_dat'])['article_id'].apply(list).reset_index()
    del df_trans
    gc.collect()

    baskets['size'] = baskets['article_id'].apply(len)
    baskets = baskets[(baskets['size'] >= 2) & (baskets['size'] <= 10)]

    print("counting co-occurrences")
    pair_counter = Counter()
    for items in baskets['article_id']:
        items = sorted(items)
        for pair in itertools.combinations(items, 2):
            pair_counter[pair] += 1

    del baskets
    gc.collect()

    base_pairs = [(k[0], k[1], v) for k, v in pair_counter.items() if v >= 3]
    df_base_graph = pd.DataFrame(base_pairs, columns=['Item_A', 'Item_B', 'weight'])
    print(f"base graph edges: {len(df_base_graph)}")

    print("applying heuristics")
    df_test = df_base_graph.merge(df_meta.add_suffix('_A'), left_on='Item_A', right_on='article_id_A', how='left')
    df_test = df_test.merge(df_meta.add_suffix('_B'), left_on='Item_B', right_on='article_id_B', how='left')

    unwanted_groups = ['Under-, Nightwear']
    rule_0 = (~df_test['garment_group_name_A'].isin(unwanted_groups)) & (~df_test['garment_group_name_B'].isin(unwanted_groups))

    rule_1 = df_test['index_name_A'] == df_test['index_name_B']

    allowed_same_groups = ['Accessories', 'Swimwear']
    rule_2 = (df_test['product_type_name_A'] != df_test['product_type_name_B']) | (df_test['product_group_name_A'] == 'Accessories')
    
    rule_3_prod = df_test['product_group_name_A'] != df_test['product_group_name_B']
    rule_3_garm = df_test['garment_group_name_A'] != df_test['garment_group_name_B']
    rule_3 = (rule_3_prod & rule_3_garm) | (df_test['product_group_name_A'].isin(allowed_same_groups))

    rule_5 = df_test['Item_A'].str[:6] != df_test['Item_B'].str[:6]

    color_a = df_test['colour_group_name_A']
    color_b = df_test['colour_group_name_B']

    neutrals = ['Black', 'White', 'Off White', 'Grey', 'Light Grey', 'Dark Grey', 'Beige', 'Light Beige', 'Greyish Beige', 'Silver', 'Gold', 'Transparent']
    navy_blues = ['Dark Blue', 'Navy']
    blues = ['Blue', 'Light Blue', 'Other Blue', 'Turquoise', 'Light Turquoise', 'Dark Turquoise']
    greens = ['Green', 'Light Green', 'Dark Green', 'Greenish Khaki', 'Other Green', 'Olive']
    pinks_reds = ['Red', 'Light Red', 'Dark Red', 'Pink', 'Light Pink', 'Dark Pink', 'Other Red', 'Other Pink', 'Burgundy']
    yellows_oranges = ['Orange', 'Light Orange', 'Dark Orange', 'Yellow', 'Light Yellow', 'Dark Yellow', 'Other Yellow', 'Other Orange', 'Bronze/Copper']
    browns = ['Brown', 'Dark Brown', 'Yellowish Brown']
    purples = ['Purple', 'Light Purple', 'Dark Purple', 'Other Purple']

    rule_6_mono = color_a == color_b
    rule_6_neutral = color_a.isin(neutrals) | color_b.isin(neutrals)
    rule_6_navy = color_a.isin(navy_blues) | color_b.isin(navy_blues)
    rule_6_denim = (df_test['garment_group_name_A'] == 'Trousers Denim') | (df_test['garment_group_name_B'] == 'Trousers Denim')
    
    rule_6_tonal_blue = color_a.isin(blues) & color_b.isin(blues)
    rule_6_tonal_green = color_a.isin(greens) & color_b.isin(greens)
    rule_6_tonal_redpink = color_a.isin(pinks_reds) & color_b.isin(pinks_reds)
    rule_6_tonal_warm = color_a.isin(yellows_oranges) & color_b.isin(yellows_oranges)
    rule_6_tonal_brown = color_a.isin(browns) & color_b.isin(browns)
    
    rule_6_cross_earth = (color_a.isin(browns) & color_b.isin(greens + yellows_oranges)) | (color_b.isin(browns) & color_a.isin(greens + yellows_oranges))
    rule_6_cross_blue_pink = (color_a.isin(blues) & color_b.isin(pinks_reds)) | (color_b.isin(blues) & color_a.isin(pinks_reds))

    rule_6 = (rule_6_mono | rule_6_neutral | rule_6_navy | rule_6_denim | 
              rule_6_tonal_blue | rule_6_tonal_green | rule_6_tonal_redpink | 
              rule_6_tonal_warm | rule_6_tonal_brown | rule_6_cross_earth | rule_6_cross_blue_pink)

    season_a = df_test['seasonality_A'].fillna('').str.lower()
    season_b = df_test['seasonality_B'].fillna('').str.lower()
    
    a_is_winter = season_a.str.contains('winter|fall|snow|cold|chill')
    a_is_summer = season_a.str.contains('summer|beach|heat|hot')
    b_is_winter = season_b.str.contains('winter|fall|snow|cold|chill')
    b_is_summer = season_b.str.contains('summer|beach|heat|hot')
    
    a_is_all_season = season_a.str.contains('all-season|all-year|any season|transition')
    b_is_all_season = season_b.str.contains('all-season|all-year|any season|transition')

    conflict_winter_summer = (a_is_winter & b_is_summer) & (~a_is_all_season) & (~b_is_all_season)
    conflict_summer_winter = (a_is_summer & b_is_winter) & (~a_is_all_season) & (~b_is_all_season)
    rule_7 = ~(conflict_winter_summer | conflict_summer_winter)

    section_a = df_test['section_name_A'].fillna('')
    section_b = df_test['section_name_B'].fillna('')
    
    sports = ['Ladies H&M Sport', 'Men H&M Sport', 'Kids Sports']
    formal = ['Womens Tailoring', 'Men Suits & Tailoring', 'Contemporary Smart']
    
    a_is_sport, b_is_sport = section_a.isin(sports), section_b.isin(sports)
    a_is_formal, b_is_formal = section_a.isin(formal), section_b.isin(formal)
    
    rule_8 = ~((a_is_sport & b_is_formal) | (a_is_formal & b_is_sport))

    final_mask = rule_0 & rule_1 & rule_2 & rule_3 & rule_5 & rule_6 & rule_7 & rule_8
    df_filtered = df_test[final_mask][['Item_A', 'Item_B', 'weight']]

    print("formatting bidirectional graph")
    edges = []
    for row in df_filtered.itertuples(index=False):
        edges.append({'item_a': row.Item_A, 'item_b': row.Item_B, 'weight': row.weight})
        edges.append({'item_a': row.Item_B, 'item_b': row.Item_A, 'weight': row.weight})

    df_final_graph = pd.DataFrame(edges)
    df_final_graph = df_final_graph.groupby(['item_a', 'item_b'])['weight'].sum().reset_index()
    df_final_graph = df_final_graph.sort_values(by=['item_a', 'weight'], ascending=[True, False])

    df_final_graph.to_csv(OUTPUT_FILE, index=False)
    
    elapsed = time.time() - start_time
    print(f"graph saved to {OUTPUT_FILE}")
    print(f"final edge count: {len(df_final_graph)}")
    print(f"execution time: {elapsed:.2f}s")

if __name__ == '__main__':
    main()