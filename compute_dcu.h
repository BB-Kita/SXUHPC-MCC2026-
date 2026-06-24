#pragma once

// Multi-DCU scheduler. h_sst_data layout: [DAYS_TOTAL][LAT_SIZE][LON_SIZE].
void dispatch_to_4_dcus(float* h_sst_data);

void dispatch_to_4_dcus_with_output(float* h_sst_data, float* h_mean, float* h_p90,
                                    const int* changed_slots = nullptr, int num_changed = 0);

// Experimental persistent-buffer scheduler: first call uploads all slots,
// later calls upload only changed_slots.
void dispatch_to_4_dcus_with_output_incremental(float* h_sst_data, float* h_mean, float* h_p90,
                                                const int* changed_slots = nullptr, int num_changed = 0);

void dispatch_to_4_dcus_with_output_incremental_hook(float* h_sst_data, float* h_src_data, float* h_mean, float* h_p90,
                                                     const int* changed_slots, int num_changed,
                                                     void (*after_upload)(void*),
                                                     void* after_upload_user);

void dispatch_source_window_preloaded(float* h_source_data, int source_days, int target_offset,
                                      float* h_mean, float* h_p90);

void dispatch_source_window_preloaded_local(float** h_source_local, int source_days, int target_offset,
                                            float* h_mean, float* h_p90);

void init_source_window_device_preloaded(int source_days);
void upload_source_days_preloaded_local(float** h_source_local, int source_day0, int source_day_count);
void upload_source_days_preloaded_local_async(float** h_source_local, int source_day0, int source_day_count);
void wait_source_days_preloaded_upload();
void dispatch_source_window_device_preloaded(int source_days, int target_offset,
                                             float* h_mean, float* h_p90);

void cleanup_dcu_persistent_buffers();
void cleanup_dcu_buffers();
