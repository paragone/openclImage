#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

const char *rearrangeKernel = R"(
__kernel void rearrange_nv21(
    __global const uchar* inputY,   
    __global const uchar* inputUV,  
    __global uchar* outputY,       
    __global uchar* outputUV,       
    int inputWidth,                
    int inputHeight)                
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int outputWidth = inputWidth / 2;   
    int outputHeight = inputHeight * 2; 

    // 处理 Y 分量
    if (y < inputHeight) {
        int srcX = x;
        int srcY = y;
        outputY[y * outputWidth + x] = inputY[srcY * inputWidth + srcX];
    } else {
        int srcX = x + outputWidth;
        int srcY = y - inputHeight;
        outputY[y * outputWidth + x] = inputY[srcY * inputWidth + srcX];
    }

    // 处理 UV 分量
    if (y % 2 == 0) {
        int uvX = x / 2;
        int uvY = y / 2;

        if (y < inputHeight) {
            int srcIndexUV = uvY * inputWidth + uvX * 2;
            int dstIndexUV = uvY * outputWidth + uvX * 2;
            
            outputUV[dstIndexUV] = inputUV[srcIndexUV];     // V 分量
            outputUV[dstIndexUV + 1] = inputUV[srcIndexUV + 1]; // U 分量
        } else {
            int srcIndexUV = (uvY - inputHeight / 2) * inputWidth + (uvX + outputWidth / 2) * 2; 
            int dstIndexUV = uvY * outputWidth + uvX * 2; 
            
            outputUV[dstIndexUV] = inputUV[srcIndexUV];     // V 分量
            outputUV[dstIndexUV + 1] = inputUV[srcIndexUV + 1]; // U 分量
        }
    }
}
)";



void rearrangeNV21(const std::string &inputFile, const std::string &outputFile, int width, int height)
{
    // Calculate sizes
    const size_t yPlaneSize = width * height;
    const size_t uvPlaneSize = width * height / 2;
    const size_t totalSize = yPlaneSize + uvPlaneSize;

    // Read input NV21 file
    std::ifstream inFile(inputFile, std::ios::binary);
    if (!inFile)
    {
        throw std::runtime_error("Failed to open input file: " + inputFile);
    }
    std::vector<unsigned char> nv21Data(totalSize);
    inFile.read(reinterpret_cast<char *>(nv21Data.data()), totalSize);
    inFile.close();

    // Split Y and UV planes
    std::vector<unsigned char> yPlane(nv21Data.begin(), nv21Data.begin() + yPlaneSize);
    std::vector<unsigned char> uvPlane(nv21Data.begin() + yPlaneSize, nv21Data.end());

    // Prepare OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    // Create command queue with profiling enabled
    // Create command queue with profiling enabled
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl::CommandQueue queue(context, device, properties);

    // Create buffers
    cl::Buffer inputYBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, yPlaneSize, yPlane.data());
    cl::Buffer outputYBuffer(context, CL_MEM_WRITE_ONLY, yPlaneSize);
    cl::Buffer inputUVBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, uvPlaneSize, uvPlane.data());
    cl::Buffer outputUVBuffer(context, CL_MEM_WRITE_ONLY, uvPlaneSize);

    cl::Program::Sources sources;
    sources.push_back({rearrangeKernel, strlen(rearrangeKernel)});
    cl::Program program(context, sources);
    program.build({device});

    cl::Kernel kernel(program, "rearrange_nv21");

    kernel.setArg(0, inputYBuffer);
    kernel.setArg(1, inputUVBuffer);
    kernel.setArg(2, outputYBuffer);
    kernel.setArg(3, outputUVBuffer);
    kernel.setArg(4, width * 2);
    kernel.setArg(5, height / 2);

    // Launch kernels and profile execution time
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange, nullptr, &event);
    queue.finish();

    // Read back results
    std::vector<unsigned char> outputYPlane(yPlaneSize);
    std::vector<unsigned char> outputUVPlane(uvPlaneSize);

    queue.enqueueReadBuffer(outputYBuffer, CL_TRUE, 0, yPlaneSize, outputYPlane.data());
    queue.enqueueReadBuffer(outputUVBuffer, CL_TRUE, 0, uvPlaneSize, outputUVPlane.data());

    // Combine Y and UV planes into output buffer
    std::vector<unsigned char> outputNV21(totalSize);
    std::copy(outputYPlane.begin(), outputYPlane.end(), outputNV21.begin());
    std::copy(outputUVPlane.begin(), outputUVPlane.end(), outputNV21.begin() + yPlaneSize);

    // Write to output NV21 file
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile)
    {
        throw std::runtime_error("Failed to create output file: " + outputFile);
    }
    outFile.write(reinterpret_cast<const char *>(outputNV21.data()), outputNV21.size());
    outFile.close();

    // Print profiling results
    cl_ulong startTime, endTime;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
    std::cout << "NV21 rearrange time: " << (endTime - startTime) / 1e6 << " ms" << std::endl;
}

int main()
{
    try
    {
        const std::string inputFile = "input.nv21";
        const std::string outputFile = "output.nv21";
        const int width = 7680 / 2;  // Split width
        const int height = 1300 * 2; // Double height after rearrange

        rearrangeNV21(inputFile, outputFile, width, height);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}