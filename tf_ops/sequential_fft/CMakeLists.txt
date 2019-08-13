

add_library(sequential_batch_fft.so sequential_batch_fft.cc sequential_batch_fft_kernel.cu.cc)
target_include_directories(sequential_batch_fft.so PRIVATE $ENV{TF_INCLUDE_PATH})
target_include_directories(sequential_batch_fft.so PUBLIC /usr/local/Cellar/eigen/3.3.7/include/eigen3)
target_link_options(sequential_batch_fft.so PRIVATE -undefined dynamic_lookup)