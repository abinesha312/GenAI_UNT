#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <limits>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << " at line "       \
                      << __LINE__ << ": " << cudaGetErrorString(err)       \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

//-----------------------------------------------------
// DynamicAllocator Class (Object-Oriented)
//-----------------------------------------------------
class DynamicAllocator {
public:
    DynamicAllocator(int device = 0) : device_(device) {
        CUDA_CHECK(cudaSetDevice(device_));
        CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool_, device_));
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    ~DynamicAllocator() {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    // Asynchronously allocate memory of given size (in bytes)
    void* allocate(size_t size) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocAsync(&ptr, size, stream_));
        return ptr;
    }
    // Asynchronously free memory
    void free(void* ptr) {
        CUDA_CHECK(cudaFreeAsync(ptr, stream_));
    }
    // Synchronize the stream to ensure all operations are complete
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    cudaStream_t getStream() const { return stream_; }
private:
    int device_;
    cudaMemPool_t memPool_;
    cudaStream_t stream_;
};

//-----------------------------------------------------
// Dummy Kernel to Simulate Inference Workload
//-----------------------------------------------------
__global__ void dummyKernel(float* data, size_t numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Simple compute-bound operation
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

//-----------------------------------------------------
// BenchmarkRunner Class
//-----------------------------------------------------
class BenchmarkRunner {
public:
    BenchmarkRunner(DynamicAllocator& allocator, size_t numElements, int iterations)
        : allocator_(allocator), numElements_(numElements), iterations_(iterations) {
        kernelTimes_.resize(iterations_, 0.0f);
        sizeInBytes_ = numElements_ * sizeof(float);
        d_data_ = static_cast<float*>(allocator_.allocate(sizeInBytes_));
        allocator_.synchronize();
        // Initialize memory (set all elements to 1)
        CUDA_CHECK(cudaMemsetAsync(d_data_, 1, sizeInBytes_, allocator_.getStream()));
        allocator_.synchronize();
    }
    ~BenchmarkRunner() {
        if (d_data_) {
            allocator_.free(d_data_);
            allocator_.synchronize();
            d_data_ = nullptr;
        }
    }
    // Run dummy kernel and record execution times using CUDA events
    void runKernel() {
        int threadsPerBlock = 256;
        int blocks = (numElements_ + threadsPerBlock - 1) / threadsPerBlock;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        for (int i = 0; i < iterations_; ++i) {
            CUDA_CHECK(cudaEventRecord(start, allocator_.getStream()));
            dummyKernel<<<blocks, threadsPerBlock, 0, allocator_.getStream()>>>(d_data_, numElements_);
            CUDA_CHECK(cudaEventRecord(stop, allocator_.getStream()));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            kernelTimes_[i] = ms;
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    double getAverageKernelTime() const {
        double sum = 0.0;
        for (float t : kernelTimes_) sum += t;
        return sum / iterations_;
    }
    double getMinKernelTime() const {
        double minTime = std::numeric_limits<double>::max();
        for (float t : kernelTimes_) {
            if (t < minTime) minTime = t;
        }
        return minTime;
    }
    double getMaxKernelTime() const {
        double maxTime = 0.0;
        for (float t : kernelTimes_) {
            if (t > maxTime) maxTime = t;
        }
        return maxTime;
    }
    double getThroughput() const {
        double avgTimeSec = getAverageKernelTime() / 1000.0;
        return numElements_ / avgTimeSec;
    }
private:
    DynamicAllocator& allocator_;
    size_t numElements_;
    int iterations_;
    size_t sizeInBytes_;
    float* d_data_;
    std::vector<float> kernelTimes_;
};

//-----------------------------------------------------
// CPU-GPU Transfer Benchmark Functions
//-----------------------------------------------------
double benchmarkHostToDevice(size_t sizeInBytes, int iterations, cudaStream_t stream) {
    std::vector<double> times;
    char* h_data = new char[sizeInBytes];
    char* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeInBytes));
    for (int i = 0; i < iterations; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, sizeInBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;
    return std::accumulate(times.begin(), times.end(), 0.0) / iterations;
}

double benchmarkDeviceToHost(size_t sizeInBytes, int iterations, cudaStream_t stream) {
    std::vector<double> times;
    char* h_data = new char[sizeInBytes];
    char* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeInBytes));
    for (int i = 0; i < iterations; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, sizeInBytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;
    return std::accumulate(times.begin(), times.end(), 0.0) / iterations;
}

//-----------------------------------------------------
// CSV Output Utility
//-----------------------------------------------------
void writeCSV(const std::string &filename, const std::vector<std::vector<std::string>> &data) {
    std::ofstream file(filename);
    for (const auto &row : data) {
        for (size_t i = 0; i < row.size(); i++) {
            file << row[i];
            if (i != row.size()-1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

//-----------------------------------------------------
// Main Function
//-----------------------------------------------------
int main() {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    DynamicAllocator allocator(device);

    // Benchmark asynchronous allocation and free times
    size_t numElements = 1 << 20; // 1 million elements
    size_t sizeInBytes = numElements * sizeof(float);
    int allocIterations = 10;
    std::vector<double> allocTimes;
    std::vector<double> freeTimes;
    for (int i = 0; i < allocIterations; ++i) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, allocator.getStream()));
        void* ptr = allocator.allocate(sizeInBytes);
        CUDA_CHECK(cudaEventRecord(stop, allocator.getStream()));
        CUDA_CHECK(cudaStreamSynchronize(allocator.getStream()));
        float ms_alloc = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_alloc, start, stop));
        allocTimes.push_back(ms_alloc);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        // Measure free time
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, allocator.getStream()));
        allocator.free(ptr);
        CUDA_CHECK(cudaEventRecord(stop, allocator.getStream()));
        CUDA_CHECK(cudaStreamSynchronize(allocator.getStream()));
        float ms_free = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_free, start, stop));
        freeTimes.push_back(ms_free);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    double avgAlloc = std::accumulate(allocTimes.begin(), allocTimes.end(), 0.0) / allocIterations;
    double avgFree  = std::accumulate(freeTimes.begin(), freeTimes.end(), 0.0) / allocIterations;
    std::cout << "Average async allocation time: " << avgAlloc << " ms" << std::endl;
    std::cout << "Average async free time: " << avgFree << " ms" << std::endl;

    // Benchmark dummy kernel execution (simulate inference)
    int kernelIterations = 20;
    BenchmarkRunner bench(allocator, numElements, kernelIterations);
    bench.runKernel();
    double avgKernelTime = bench.getAverageKernelTime();
    double minKernelTime = bench.getMinKernelTime();
    double maxKernelTime = bench.getMaxKernelTime();
    double throughput = bench.getThroughput();
    std::cout << "Kernel: Avg = " << avgKernelTime << " ms, Min = " << minKernelTime
              << " ms, Max = " << maxKernelTime << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " elements/second" << std::endl;

    // Benchmark CPU-to-GPU and GPU-to-CPU transfers
    int transferIterations = 10;
    double h2dTime = benchmarkHostToDevice(sizeInBytes, transferIterations, allocator.getStream());
    double d2hTime = benchmarkDeviceToHost(sizeInBytes, transferIterations, allocator.getStream());
    double h2dBandwidth = (sizeInBytes / (1024.0*1024.0)) / (h2dTime/1000.0); // MB/s
    double d2hBandwidth = (sizeInBytes / (1024.0*1024.0)) / (d2hTime/1000.0); // MB/s
    std::cout << "CPU-to-GPU transfer: " << h2dTime << " ms, Bandwidth: " << h2dBandwidth << " MB/s" << std::endl;
    std::cout << "GPU-to-CPU transfer: " << d2hTime << " ms, Bandwidth: " << d2hBandwidth << " MB/s" << std::endl;

    // Write CSV file for all metrics
    std::vector<std::vector<std::string>> csvData;
    csvData.push_back({"Metric", "Value"});
    csvData.push_back({"Avg Async Allocation Time (ms)", std::to_string(avgAlloc)});
    csvData.push_back({"Avg Async Free Time (ms)", std::to_string(avgFree)});
    csvData.push_back({"Avg Kernel Time (ms)", std::to_string(avgKernelTime)});
    csvData.push_back({"Min Kernel Time (ms)", std::to_string(minKernelTime)});
    csvData.push_back({"Max Kernel Time (ms)", std::to_string(maxKernelTime)});
    csvData.push_back({"Throughput (elements/sec)", std::to_string(throughput)});
    csvData.push_back({"Avg H2D Transfer Time (ms)", std::to_string(h2dTime)});
    csvData.push_back({"Avg D2H Transfer Time (ms)", std::to_string(d2hTime)});
    csvData.push_back({"H2D Bandwidth (MB/s)", std::to_string(h2dBandwidth)});
    csvData.push_back({"D2H Bandwidth (MB/s)", std::to_string(d2hBandwidth)});
    writeCSV("benchmark_summary.csv", csvData);
    std::cout << "Benchmark results written to benchmark_summary.csv" << std::endl;

    // Report GPU memory info
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "GPU Memory: " << freeMem/(1024*1024) << " MB free out of "
              << totalMem/(1024*1024) << " MB total." << std::endl;

    return 0;
}
