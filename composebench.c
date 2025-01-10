#include <iostream>
#include <vector>
#include <fstream>
#include <CL/cl.h>
#include <chrono>

// 检查 OpenCL 错误
#define CHECK_CL_ERROR(err)                                     \
    if (err != CL_SUCCESS)                                      \
    {                                                           \
        std::cerr << "OpenCL error: " << err << " at line " << __LINE__ << std::endl; \
        exit(1);                                                \
    }

// 读取文件内容（返回 std::string）
std::string readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// 写入文件内容（处理二进制数据）
void writeFile(const std::string &filename, const std::vector<unsigned char> &data)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
}

int main()
{
    // 初始化 OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // 获取平台和设备
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL_ERROR(err);

    // 创建上下文和命令队列
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err);
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    CHECK_CL_ERROR(err);

    // 读取内核文件
    std::string kernelSource = readFile("compose_nv21_buffer.cl");
    const char *kernelSourcePtr = kernelSource.c_str();

    // 创建程序对象
    program = clCreateProgramWithSource(context, 1, &kernelSourcePtr, NULL, &err);
    CHECK_CL_ERROR(err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        std::cerr << "Build log: " << buildLog << std::endl;
        exit(1);
    }

    // 创建内核对象
    kernel = clCreateKernel(program, "compose_nv21_buffer", &err);
    CHECK_CL_ERROR(err);

    // 输入图像尺寸
    int width1 = 960;  // 第一张图像的宽度
    int height = 540;  // 图像高度
    int width2 = 960;  // 第二张图像的宽度
    int output_width = width1 + width2; // 输出图像的宽度

    // 读取 NV21 数据
    std::string nv21_string1 = readFile("input1.nv21");
    std::vector<unsigned char> nv21_data1(nv21_string1.begin(), nv21_string1.end());
    std::string nv21_string2 = readFile("input1.nv21");
    std::vector<unsigned char> nv21_data2(nv21_string2.begin(), nv21_string2.end());

    // 拆分 Y 和 UV 数据
    std::vector<unsigned char> y_data1(nv21_data1.begin(), nv21_data1.begin() + width1 * height);
    std::vector<unsigned char> uv_data1(nv21_data1.begin() + width1 * height, nv21_data1.end());
    std::vector<unsigned char> y_data2(nv21_data2.begin(), nv21_data2.begin() + width2 * height);
    std::vector<unsigned char> uv_data2(nv21_data2.begin() + width2 * height, nv21_data2.end());

    // 创建输入缓冲区
    cl_mem input_y1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, y_data1.size(), y_data1.data(), &err);
    CHECK_CL_ERROR(err);
    cl_mem input_uv1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, uv_data1.size(), uv_data1.data(), &err);
    CHECK_CL_ERROR(err);
    cl_mem input_y2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, y_data2.size(), y_data2.data(), &err);
    CHECK_CL_ERROR(err);
    cl_mem input_uv2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, uv_data2.size(), uv_data2.data(), &err);
    CHECK_CL_ERROR(err);

    // 创建输出缓冲区
    std::vector<unsigned char> output_y_data(output_width * height);
    std::vector<unsigned char> output_uv_data(output_width * height / 2);

    cl_mem output_y = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_y_data.size(), NULL, &err);
    CHECK_CL_ERROR(err);
    cl_mem output_uv = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_uv_data.size(), NULL, &err);
    CHECK_CL_ERROR(err);

    // 设置内核参数
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_y1);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_uv1);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_y2);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input_uv2);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &output_y);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_uv);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &width1);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 7, sizeof(int), &width2);
    CHECK_CL_ERROR(err);
    err = clSetKernelArg(kernel, 8, sizeof(int), &height);
    CHECK_CL_ERROR(err);

   // 基准测试：运行 1000 次
    const int num_runs = 1000;
    cl_ulong total_time = 0; // 总时间（纳秒）

    for (int i = 0; i < num_runs; ++i)
    {
        cl_event event;
        size_t global_size[2] = {static_cast<size_t>(output_width), static_cast<size_t>(height)};

        // 运行内核
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, &event);
        CHECK_CL_ERROR(err);

        // 等待内核执行完成
        clFinish(queue);

        // 获取内核执行时间
        cl_ulong start_time, end_time;
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        CHECK_CL_ERROR(err);
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
        CHECK_CL_ERROR(err);

        total_time += (end_time - start_time);

        // 释放事件对象
        clReleaseEvent(event);
    }
        // 输出基准测试结果
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "Total time for " << num_runs << " runs: " << total_time / 1e6 << " ms" << std::endl;
    std::cout << "Average time per run: " << (total_time / num_runs) / 1e6 << " ms" << std::endl;

    // 读取输出数据
    err = clEnqueueReadBuffer(queue, output_y, CL_TRUE, 0, output_y_data.size(), output_y_data.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(err);
    err = clEnqueueReadBuffer(queue, output_uv, CL_TRUE, 0, output_uv_data.size(), output_uv_data.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(err);

    // 合并 Y 和 UV 数据
    std::vector<unsigned char> output_nv21_data(output_y_data.begin(), output_y_data.end());
    output_nv21_data.insert(output_nv21_data.end(), output_uv_data.begin(), output_uv_data.end());

    // 写入输出文件
    writeFile("output.nv21", output_nv21_data);

    // 释放资源
    clReleaseMemObject(input_y1);
    clReleaseMemObject(input_uv1);
    clReleaseMemObject(input_y2);
    clReleaseMemObject(input_uv2);
    clReleaseMemObject(output_y);
    clReleaseMemObject(output_uv);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Buffer-based compose completed successfully!" << std::endl;
    return 0;
}