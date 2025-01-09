#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.h>

// 检查 OpenCL 错误并打印
#define CHECK_OPENCL_ERROR(call)                                                                                                                 \
    do                                                                                                                                           \
    {                                                                                                                                            \
        cl_int err = (call);                                                                                                                     \
        if (err != CL_SUCCESS)                                                                                                                   \
        {                                                                                                                                        \
            std::cerr << "OpenCL error in " << __FILE__ << ":" << __LINE__ << " - " << #call << " failed with error code: " << err << std::endl; \
            exit(1);                                                                                                                             \
        }                                                                                                                                        \
    } while (0)

std::vector<uint8_t> readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary); // 以二进制模式打开文件
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), size);
    return buffer;
}
// 读取文件到缓冲区
// 加载 OpenCL kernel 文件
char *load_kernel_source(const char *filename, size_t *size)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *source = (char *)malloc(*size + 1);
    fread(source, 1, *size, file);
    source[*size] = '\0';
    fclose(file);
    return source;
}

// 将缓冲区写入文件
void writeFile(const std::string &filename, const std::vector<uint8_t> &buffer)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
}

int main()
{
    // 输入输出文件路径
    const std::string input_filename = "input.nv21";
    const std::string output_filename = "output.nv21";

    // 图像参数
    const int input_width = 7680;
    const int input_height = 1300;
    const int output_width = 8000;
    const int output_height = 1500;

    // 读取输入 NV21 数据
    std::vector<uint8_t> input_data = readFile(input_filename);

    // 分配输出 NV21 数据缓冲区
    std::vector<uint8_t> output_data(output_width * output_height * 3 / 2, 0);

    // OpenCL 变量
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input_buffer, output_buffer;

    // 获取平台
    CHECK_OPENCL_ERROR(clGetPlatformIDs(1, &platform, nullptr));

    // 获取设备
    CHECK_OPENCL_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    // 创建上下文
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    if (!context)
    {
        std::cerr << "Failed to create OpenCL context" << std::endl;
        exit(1);
    }

    // 创建命令队列
    queue = clCreateCommandQueue(context, device, 0, nullptr);
    if (!queue)
    {
        std::cerr << "Failed to create OpenCL command queue" << std::endl;
        exit(1);
    }

    // 创建输入缓冲区
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, input_data.size(), nullptr, nullptr);
    if (!input_buffer)
    {
        std::cerr << "Failed to create input buffer" << std::endl;
        exit(1);
    }

    // 创建输出缓冲区
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_data.size(), nullptr, nullptr);
    if (!output_buffer)
    {
        std::cerr << "Failed to create output buffer" << std::endl;
        exit(1);
    }

    // 将输入数据复制到设备
    CHECK_OPENCL_ERROR(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, input_data.size(), input_data.data(), 0, nullptr, nullptr));

    // 读取内核文件
    size_t kernel_size;
    char *kernel_source = load_kernel_source("rearrange.cl", &kernel_size);

    // printf("kernel  = %s", kernel_source_ptr);

    // 创建程序
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, nullptr, nullptr);
    if (!program)
    {
        std::cerr << "Failed to create OpenCL program" << std::endl;
        exit(1);
    }

    // 编译程序
    cl_int build_status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (build_status != CL_SUCCESS)
    {
        std::cerr << "Failed to build OpenCL program" << std::endl;
        // 获取编译日志
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log: " << log.data() << std::endl;
        exit(1);
    }

    // 创建内核
    kernel = clCreateKernel(program, "rearrange_nv21", nullptr);
    if (!kernel)
    {
        std::cerr << "Failed to create OpenCL kernel" << std::endl;
        exit(1);
    }

    // 设置内核参数
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 2, sizeof(int), &input_width));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 3, sizeof(int), &input_height));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 4, sizeof(int), &output_width));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 5, sizeof(int), &output_height));

    // 执行内核
    size_t global_work_size[2] = {static_cast<size_t>(output_width), static_cast<size_t>(output_height)};
    CHECK_OPENCL_ERROR(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

    // 将输出数据复制回主机
    CHECK_OPENCL_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, output_data.size(), output_data.data(), 0, nullptr, nullptr));
    CHECK_OPENCL_ERROR(clFinish(queue));
    // 保存输出数据到文件
    writeFile(output_filename, output_data);

    // 释放资源
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Processing completed. Output saved to " << output_filename << std::endl;
    return 0;
}