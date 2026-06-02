import numpy as np
from netCDF4 import Dataset
import os

ref_dir = "/public/home/achwjznh4b/ERA5/Climatology/"
our_dir = "/public/home/fujiake/fjk/MCC26_SXU/output/"

rmse_clim = []
rmse_p90 = []
days = []

for month in range(6, 9):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for day in range(1, days_in_month[month] + 1):
        fname = "%02d%02d.nc" % (month, day)
        ref_file = os.path.join(ref_dir, fname)
        our_file = os.path.join(our_dir, fname)

        if not os.path.exists(ref_file):
            print("WARNING: reference file missing: %s" % fname)
            continue
        if not os.path.exists(our_file):
            print("WARNING: our file missing: %s" % fname)
            continue

        with Dataset(ref_file) as ds:
            ref_clim = np.array(ds.variables["Climmean"][:])
            ref_p90 = np.array(ds.variables["P90_sst"][:])
        with Dataset(our_file) as ds:
            our_clim = np.array(ds.variables["Climmean"][:])
            our_p90 = np.array(ds.variables["P90_sst"][:])

        diff_clim = ref_clim - our_clim
        diff_p90 = ref_p90 - our_p90

        r_clim = np.sqrt(np.nanmean(diff_clim ** 2))
        r_p90 = np.sqrt(np.nanmean(diff_p90 ** 2))

        rmse_clim.append(r_clim)
        rmse_p90.append(r_p90)
        days.append(fname)

rmse_clim = np.array(rmse_clim)
rmse_p90 = np.array(rmse_p90)

print("=" * 50)
print("       6~8 RMSE Verification")
print("=" * 50)
print("Clim  avg RMSE : %.4f C" % np.nanmean(rmse_clim))
print("Clim  max RMSE : %.4f C" % np.nanmax(rmse_clim))
print("-" * 50)
print("P90   avg RMSE : %.4f C" % np.nanmean(rmse_p90))
print("P90   max RMSE : %.4f C" % np.nanmax(rmse_p90))
print("=" * 50)

avg_clim = np.nanmean(rmse_clim)
max_clim = np.nanmax(rmse_clim)
avg_p90 = np.nanmean(rmse_p90)
max_p90 = np.nanmax(rmse_p90)

print()
if max_clim < 1.0 and max_p90 < 1.0:
    print("PASS: daily max error < 1C")
else:
    print("FAIL: daily max error clim=%.4f p90=%.4f (require < 1C)" % (max_clim, max_p90))

if avg_clim < 2.0 and avg_p90 < 2.0:
    print("PASS: seasonal avg error < 2C")
else:
    print("FAIL: seasonal avg error clim=%.4f p90=%.4f (require < 2C)" % (avg_clim, avg_p90))

worst_clim_idx = np.argmax(rmse_clim)
worst_p90_idx = np.argmax(rmse_p90)
print("\nWorst clim day: %s (%.4f C)" % (days[worst_clim_idx], rmse_clim[worst_clim_idx]))
print("Worst P90  day: %s (%.4f C)" % (days[worst_p90_idx], rmse_p90[worst_p90_idx]))
