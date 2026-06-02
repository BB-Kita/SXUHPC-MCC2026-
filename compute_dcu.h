#pragma once

// Multi-DCU scheduler. h_sst_data layout: [DAYS_TOTAL][LAT_SIZE][LON_SIZE].
void dispatch_to_4_dcus(float* h_sst_data);

void dispatch_to_4_dcus_with_output(float* h_sst_data, float* h_mean, float* h_p90,
                                    const int* changed_slots = nullptr, int num_changed = 0);

// Experimental persistent-buffer scheduler: first call uploads all slots,
// later calls upload only changed_slots.
void dispatch_to_4_dcus_with_output_incremental(float* h_sst_data, float* h_mean, float* h_p90,
                                                const int* changed_slots = nullptr, int num_changed = 0);

void cleanup_dcu_persistent_buffers();
void cleanup_dcu_buffers();
