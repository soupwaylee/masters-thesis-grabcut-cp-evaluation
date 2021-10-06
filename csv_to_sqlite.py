import sqlite3
import pandas as pd

PATH_INTERACTIONS = 'postgres/20211004-193347_grab_cut_interaction.csv'
PATH_MASKS = 'postgres/20211004-193347_grab_cut_mask.csv'

IMG_SHAPE = (384, 512)


# Read from CSV, drop some rows from testing phase and format times and column names
interactions_df = pd.read_csv(PATH_INTERACTIONS, skiprows=[1, 2, 3, 4, 5, 6, 7, 8])
interactions_df['first_interaction_time'] = pd.to_datetime(interactions_df['first_interaction_time'], utc=True)
interactions_df['submission_time'] = pd.to_datetime(interactions_df['submission_time'], utc=True)
interactions_df = interactions_df.rename(columns={'id': 'interaction_uuid'})

print(f"[*] {len(interactions_df)} segmentation requests have been made.")

# Read from CSV, format column name
masks_df = pd.read_csv(PATH_MASKS)
masks_df = masks_df.rename(columns={'id': 'mask_uuid'})

print(f"[*] {len(masks_df)} masks have been submitted.")

# Write back to db
conn = sqlite3.connect('grabcutstudy.db')
print(f"[*] Connected to DB.")
interactions_df.to_sql(name='interactions', con=conn, index=False, if_exists='replace')
masks_df.to_sql(name='masks', con=conn, index=False, if_exists='replace')
print(f"[*] Wrote to DB.")
conn.close()
