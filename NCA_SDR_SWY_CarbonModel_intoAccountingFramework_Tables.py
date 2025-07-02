#THIS SCRIPT WAS DEVELOPED BY EVARISTE RUTEBUKA -NATURAL CAPITAL INSIGHTS, FOR NATURAL CAPITAL ACCOUNTING (NCA) PROJECTS OF ETHIOPIA, JUNE 2025 
##This script suite supports the transformation of InVEST model outputs into formats suitable for use in Natural Capital Accounting (NCA) 
# frameworks, specifically aligned with the System of Environmental-Economic Accounting – Ecosystem Accounting (SEEA EA) standards.
#To transform outputs from InVEST ecosystem service models—namely SDR, SWY, and Carbon Storage—into spatially explicit, 
# additions and reductions, avoided quickflows and others physically meaningful metrics (e.g., tonnes or cubic meters per pixel), 
# and summarize them in ways that are directly usable for SEEA EA ecosystem service supply tables and condition accounts.
#%%
# -------------------------------
# 1. PER-PIXEL UNIT CONVERSION MODULES
# -------------------------------

# -------------------------------
# === A.Python Script: Convert SDR Rasters to Tonnes per Pixel when  SDR are from InVEST 3.16 version ===
# -------------------------------
# This script processes SDR rasters to convert values to tonnes per pixel

import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import mapping
from datetime import datetime
from tqdm import tqdm
import time
import warnings
#%%
# Define input and outputs as same folder (where your SDR outputs rasters are stored)
input_folder = Path("/SDR_Ouputs")
# Conversion factor for 90m x 90m pixel (0.81 hectares) //This is 90m is related to the DEM resolution
conversion_factor = 0.81

# Loop through all .tif files
for input_path in input_folder.glob("*.tif"):
    # Define output path with "_TperPixel" suffix
    output_path = input_path.with_name(input_path.stem + "_TperPixel.tif")

    print(f"Processing: {input_path.name}")

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update({
            'compress': 'zstd',
            'zstd_level': 9,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'dtype': 'float32',  # Preserve decimal precision
            'nodata': src.nodata if src.nodata is not None else -9999
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            for ji, window in src.block_windows(1):
                data = src.read(1, window=window).astype(np.float32)

                # Apply conversion to valid pixels only
                if profile['nodata'] is not None:
                    mask = data != profile['nodata']
                    data[mask] *= conversion_factor
                else:
                    data *= conversion_factor

                dst.write(data, 1, window=window)

    print(f"Saved: {output_path.name}")

# %%
# -------------------------------
# === B. Python Script: Convert SWY Rasters to Cubic Meters per Pixel 
# -------------------------------
# Define input folder (where your precipitation rasters are stored)
input_folder = Path("SWY_Ouputs")
# Conversion factor: mm × 8.1 = m³ per pixel (for 90m x 90m)// This 90m is from the DEM resolution
conversion_factor = 8.1

# Loop through all .tif files in the folder
for input_path in input_folder.glob("*.tif"):
    output_path = input_path.with_name(input_path.stem + "_CubicMeterPerPixel.tif")

    print(f"Processing: {input_path.name}")

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update({
            'compress': 'zstd',
            'zstd_level': 9,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'dtype': 'float32',  # to preserve decimal volumes
            'nodata': src.nodata if src.nodata is not None else -9999
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            for ji, window in src.block_windows(1):
                data = src.read(1, window=window).astype(np.float32)

                # Apply conversion to valid pixels
                if profile['nodata'] is not None:
                    mask = data != profile['nodata']
                    data[mask] *= conversion_factor
                else:
                    data *= conversion_factor

                dst.write(data, 1, window=window)

    print(f"Saved: {output_path.name}")


#%%
# -------------------------------
# C. Python Script: Convert Carbon Rasters to Tonnes of Carbon per Pixel
# -------------------------------
# This script processes Carbon rasters to convert values to tonnes of carbon per pixel
#%% Input folder
input_folder = Path("/Carbon_Outputs")
# Conversion factor for 15m x 15m pixel (225 m² = 0.0225 ha)// This 15m is from the LULC resolution
conversion_factor = 0.0225

# ✅ Add this block right here
all_files = list(input_folder.glob("*.tif")) + list(input_folder.glob("*.if"))
print(f"📂 Found {len(all_files)} raster files.")
if not all_files:
    print("❌ No raster files found. Check file names or extension.")

# ✅ Then use the found files here:
for input_path in all_files:
    output_path = input_path.with_name(input_path.stem + "_TonOfC_perPixel.tif")
    print(f"🔄 Processing: {input_path.name}")

# Loop through raster files
for input_path in input_folder.glob("*.tif"):
    output_path = input_path.with_name(input_path.stem + "_TonOfC_perPixel.tif")
    print(f"🔄 Processing: {input_path.name}")

    with rasterio.open(input_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999
        profile = src.profile.copy()
        profile.update({
            'dtype': 'float32',
            'nodata': nodata,
            'compress': 'zstd',
            'zstd_level': 9,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            for ji, window in src.block_windows(1):
                data = src.read(1, window=window).astype(np.float32)

                mask = data != nodata
                data[mask] *= conversion_factor

                dst.write(data, 1, window=window)

    print(f"✅ Saved: {output_path.name}")

# %% 
# -------------------------------
# 2. ZONAL STATISTICS BY LULC
# -------------------------------
# # === A. Python Script: Zonal statistics for SDR, SWY and Carbon Model Outputs by LULC I=(ECosystem type)
# -------------------------------
# This batch LULC Zonal Summary with Low RAM Usage (Windowed Reading)
#%%
# ✅ Start timer
start_time = time.time()
print(f"🚀 Started at {time.strftime('%H:%M:%S')}")
# ✅ Define paths
lulc_path = Path("/LULC_YearY.tif")
model_folder = Path("/SDR_SWY_CARBON_Outputs_PerPixel_YearY")
# ✅ LULC code-to-class name mapping
lulc_classes = {
    20: "Bare land",
    4:  "Annual crop",
    3:  "Perennial crop",
    11: "Dense forest",
    2:  "Moderate forest",
    12: "Sparse forest",
    8:  "Open grass land",
    10: "Closed grass land",
    16: "Open shrub land",
    18: "Closed shrub land",
    15: "Built up area",
    19: "Water body",
    13: "Wet land"
}

# ✅ Identify unique LULC codes
with rasterio.open(lulc_path) as lulc_src:
    lulc_nodata = lulc_src.nodata
    unique_codes = set()
    for _, window in lulc_src.block_windows():
        block = lulc_src.read(1, window=window)
        unique_codes.update(np.unique(block))

if lulc_nodata is not None:
    unique_codes.discard(lulc_nodata)
unique_codes = sorted(unique_codes)

# ✅ Initialize results table
results = {
    "LULC_Code": unique_codes,
    "Class_Name": [lulc_classes.get(code, f"Class_{code}") for code in unique_codes]
}

# ✅ Process each Carbon/SDR/SWY raster
for raster_path in model_folder.glob("*.tif"):
    if "LULC" in raster_path.stem:
        continue  # skip LULC itself

    column_name = f"{raster_path.stem}_YearY"
    print(f"📊 Processing: {raster_path.name}")

    # Initialize per-class totals
    totals_by_code = {code: 0.0 for code in unique_codes}

    with rasterio.open(lulc_path) as lulc_src, rasterio.open(raster_path) as val_src:
        assert lulc_src.shape == val_src.shape, f"Shape mismatch: {lulc_path.name} vs {raster_path.name}"

        lulc_windows = lulc_src.block_windows()
        lulc_nodata = lulc_src.nodata
        val_nodata = val_src.nodata

        for _, window in lulc_windows:
            lulc_block = lulc_src.read(1, window=window)
            val_block = val_src.read(1, window=window)

            for code in unique_codes:
                mask = (lulc_block == code)
                if val_nodata is not None:
                    values = np.where((mask) & (val_block != val_nodata), val_block, np.nan)
                else:
                    values = np.where(mask, val_block, np.nan)
                totals_by_code[code] += np.nansum(values)

    # Add results to main dictionary
    results[column_name] = [totals_by_code[code] for code in unique_codes]

# ✅ Save to CSV
df = pd.DataFrame(results)
output_csv = model_folder / "_zonal_totals_by_LULC_YearY.csv"
df.to_csv(output_csv, index=False)
print(f"✅ Saved to: {output_csv}")
# ✅ Print completion time
end_time = time.time()
elapsed = end_time - start_time
minutes = int(elapsed // 60)
seconds = int(elapsed % 60)
print(f"✅ Finished at {time.strftime('%H:%M:%S')}")
print(f"🕒 Total run time: {minutes} minutes and {seconds} seconds")

# %%
# -------------------------------
## ===  CARBON STOCK: Reductions and Additions
# -------------------------------
# This script processes carbon storage rasters to compute additions and reductions based on LULC changes
# Efficient Block-Wise Raster Processing with Compression
# This script processes carbon storage rasters to compute additions and reductions based on LULC changes

#%%
# Timestamp: Start
start_time = datetime.now()
print(f"🔄 Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Input paths
carbon_2013_path = r'\Carbon_TonOfC_perPixel_YearX.tif'
carbon_2022_path = r'\Carbon_TonOfC_perPixel_YearY.tif'
lulc_2013_path = r'\LULC_YearX.tif'
lulc_2022_path = r'\LULC_YearY.tif'

additions_path = r'\Carbon_additions.tif'
reductions_path = r'\Carbon_reductions.tif'

# Open input rasters
with rasterio.open(carbon_2013_path) as c13, \
     rasterio.open(carbon_2022_path) as c22, \
     rasterio.open(lulc_2013_path) as l13, \
     rasterio.open(lulc_2022_path) as l22:

    profile = c13.profile.copy()
    profile.update({
        'dtype': 'float32',
        'compress': 'deflate',
        'predictor': 2
    })

    # Prepare output files
    with rasterio.open(additions_path, 'w', **profile) as add_out, \
         rasterio.open(reductions_path, 'w', **profile) as red_out:

        # Count total windows
        total_windows = sum(1 for _ in c13.block_windows(1))

        # Process each block with progress bar
        for ji, window in tqdm(c13.block_windows(1), total=total_windows, desc="📦 Processing blocks"):
            c13_data = c13.read(1, window=window)
            c22_data = c22.read(1, window=window)
            l13_data = l13.read(1, window=window)
            l22_data = l22.read(1, window=window)

            delta = c22_data - c13_data
            lulc_changed = (l13_data != l22_data)

            additions = np.where(delta > 0, delta, 0).astype('float32')
            reductions = np.where(delta < 0, -delta, 0).astype('float32')

            add_out.write(additions, 1, window=window)
            red_out.write(reductions, 1, window=window)

# Timestamp: End
end_time = datetime.now()
elapsed = end_time - start_time
print(f"✅ Process completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Total processing time: {elapsed}")

# %%

## Dealing with the Avoided Quick Flow concept for flooding reduction
# %%
# %% Dealing with the Avoided Quick Flow concept for flood reduction
# Python Script: Calculate Avoided Quick Flow (QF) for Flood Reduction

# This script calculates the avoided Quick Flow (QF) for flood reduction based on LULC changes  
# === CONFIGURATION ===
folder = Path("/SDR_outputs")

qf_bare_path = folder / "QF_barelandScen.tif"  # this is the QF bareland scenario raster
qf_2013_path = folder / "QF_SWY_yearX_CubicMeterPerPixel.tif"
qf_2022_path = folder / "QF_SWY_yearY_CubicMeterPerPixel.tif"
lulc_2013_path = folder / "LULC_YearX.tif"
lulc_2022_path = folder / "LULC_YearY.tif"

avoid_qf_2013_path = folder / "Avoided_QF_2013.tif"
avoid_qf_2022_path = folder / "Avoided_QF_2022.tif"

conversion_factor = 8.1  # mm to m³/pixel # (for 90m x 90m pixel size of DEM used in SWY model)

lulc_classes = {
    20: "Bare land", 4: "Annual crop", 3: "Perennial crop", 11: "Dense forest",
    2: "Moderate forest", 12: "Sparse forest", 8: "Open grass land",
    10: "Closed grass land", 16: "Open shrub land", 18: "Closed shrub land",
    15: "Built up area", 19: "Water body", 13: "Wet land"
}

# === TIMESTAMP START ===
start = datetime.now()
print(f"🚀 Started at: {start.strftime('%Y-%m-%d %H:%M:%S')}")

# === STEP 1: Convert QF_bareland to m³ ===
print("🔁 Converting QF_bareland to cubic meters...")
with rasterio.open(qf_bare_path) as src:
    qf_bare = src.read(1, masked=True).astype(np.float32)
    qf_bare_cubic = qf_bare * conversion_factor
    profile = src.profile.copy()

# === STEP 2: Load Actual QF ===
def load_raster_array(path, label=""):
    with rasterio.open(path) as src:
        print(f"📥 Loading {label}: {path.name}")
        arr = src.read(1, masked=True).astype(np.float32)
    return arr

qf_2013 = load_raster_array(qf_2013_path, "QF_2013")
qf_2022 = load_raster_array(qf_2022_path, "QF_2022")

# === Check shape consistency ===
assert qf_bare_cubic.shape == qf_2013.shape == qf_2022.shape, "❌ Raster shape mismatch!"

# === STEP 3: Calculate Avoided QF ===
print("➗ Calculating avoided Quick Flow...")
avoided_qf_2013 = qf_bare_cubic - qf_2013
avoided_qf_2022 = qf_bare_cubic - qf_2022

# === STEP 4: Save Avoided QF Rasters ===
def save_raster(path, array, profile, label=""):
    profile.update(dtype='float32', nodata=-9999, compress='deflate')
    array_filled = np.where(array.mask, -9999, array.filled(0)).astype(np.float32)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array_filled, 1)
    print(f"💾 Saved: {label} -> {path.name}")

save_raster(avoid_qf_2013_path, avoided_qf_2013, profile, "Avoided QF 2013")
save_raster(avoid_qf_2022_path, avoided_qf_2022, profile, "Avoided QF 2022")

# === STEP 5: Zonal Statistics ===
def zonal_sum(value_raster_path, lulc_path):
    with rasterio.open(value_raster_path) as val_src, rasterio.open(lulc_path) as lulc_src:
        val = val_src.read(1, masked=True).astype(np.float32)
        lulc = lulc_src.read(1).astype(np.int32)

        val_array = val.filled(np.nan)
        lulc[lulc == lulc_src.nodata] = -1

        unique_codes = sorted(np.unique(lulc))
        result = {}
        print(f"🔎 Calculating zonal sum for: {value_raster_path.name}")
        for code in tqdm(unique_codes, desc="Zonal summary", ncols=80):
            if code == -1:
                continue
            mask = (lulc == code)
            result[code] = np.nansum(val_array[mask])
    return result

zonal_2013 = zonal_sum(avoid_qf_2013_path, lulc_2013_path)
zonal_2022 = zonal_sum(avoid_qf_2022_path, lulc_2022_path)

# === STEP 6: Save Summary Table ===
sorted_codes = sorted(zonal_2013.keys())
df = pd.DataFrame({
    "LULC_Code": sorted_codes,
    "Class_Name": [lulc_classes.get(c, f"Class_{c}") for c in sorted_codes],
    "Avoided_QF_2013_m3": [zonal_2013[c] for c in sorted_codes],
    "Avoided_QF_2022_m3": [zonal_2022.get(c, 0.0) for c in sorted_codes]
})

output_table = folder / "Avoided_QuickFlow_Zonal_Stats.csv"
df.to_csv(output_table, index=False)
print(f"✅ Zonal stats saved to: {output_table.name}")

# === TIMESTAMP END ===
end = datetime.now()
elapsed = end - start
print(f"✅ Finished at: {end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Total elapsed time: {elapsed}")

#%%
# Python Script: Zonal Statistics for Ecosystem Services by Watershed or province
# This script processes raster data to calculate zonal statistics for ecosystem services by watershed
# It reads watershed boundaries from a GeoPackage, processes multiple raster files, and outputs results to a CSV file.
# %%


gpkg_path = Path(r"/Watershed.gpkg")  # or Province.gpkg
raster_folder = Path(r"/SDR_SWY_CARBON_Outputs_PerPixel_YearY")
output_csv = Path(r"/EcosServices_sum_by_watershed.csv")

# --- LOAD VECTOR ---
watersheds = gpd.read_file(gpkg_path)
watersheds = watersheds.to_crs("EPSG:4326")  # Adjust to match raster CRS if needed

# --- OUTPUT DATAFRAME ---
final_df = watersheds[["Watershed", "ws_id", "Area_ha"]].copy() # this reflect the existing field of shaepefile

# --- START TIMER ---
global_start = time.time()

# --- RASTER LOOP ---
raster_list = list(raster_folder.glob("*.tif"))
total_rasters = len(raster_list)

for idx_r, raster_path in enumerate(raster_list, start=1):
    raster_name = raster_path.stem
    print(f"🟡 [{idx_r}/{total_rasters}] Processing: {raster_name} ({(idx_r/total_rasters)*100:.1f}%)")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        watersheds_proj = watersheds.to_crs(raster_crs)

        zonal_sums = []
        total_zones = len(watersheds_proj)

        for i, geom in enumerate(watersheds_proj.geometry, start=1):
            try:
                out_image, _ = rasterio.mask.mask(src, [mapping(geom)], crop=True)
                out_image = out_image[0]

                nodata = src.nodata
                if nodata is not None:
                    valid_mask = out_image != nodata
                else:
                    valid_mask = out_image > -1e+30

                sum_val = np.nansum(out_image[valid_mask])

            except Exception as e:
                warnings.warn(f"⚠️ Geometry {i} failed: {e}")
                sum_val = np.nan

            zonal_sums.append(sum_val)

            # Optional per-zone progress
            if total_zones >= 10 and i % (total_zones // 10) == 0:
                pct = (i / total_zones) * 100
                print(f"   └─ Watersheds done: {i}/{total_zones} ({pct:.0f}%)")

        final_df[raster_name] = zonal_sums

# --- SAVE CSV ---
final_df.to_csv(output_csv, index=False)

# --- DONE ---
total_time = time.time() - global_start
print(f"\n✅ All done. Output saved to:\n{output_csv}")
print(f"⏱️ Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

