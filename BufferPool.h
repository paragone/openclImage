#pragma once

#include <memory>
#include <CL/cl.h>
#include <variant>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <tuple>
#include <sys/mman.h>

namespace bos::mm
{
    // Buffer 类
    class Buffer
    {
    public:
        enum class Type
        {
            NORMAL,  // 普通内存
            DMABUF   // DMA缓冲区
        };

        // 构造函数
        Buffer(Type type, size_t size, int dma_fd = -1)
            : type(type), size(size), dma_fd(dma_fd), data(nullptr)
        {
            if (type == Type::NORMAL)
            {
                data = std::make_unique<uint8_t[]>(size); // 普通内存分配
            }
            else if (type == Type::DMABUF && dma_fd != -1)
            {
                // 映射DMA文件描述符到内存
                data = static_cast<uint8_t *>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dma_fd, 0));
                if (data == MAP_FAILED)
                {
                    throw std::runtime_error("Failed to map DMA buffer to memory.");
                }
            }
            else
            {
                throw std::invalid_argument("Invalid parameters for DMA buffer.");
            }
        }

        // 析构函数
        virtual ~Buffer()
        {
            if (type == Type::DMABUF && data != nullptr)
            {
                // 解除映射
                munmap(data, size);
            }
        }

        // 获取数据指针
        uint8_t *get_data() { return data; }

        // 获取缓冲区类型
        Type get_type() const { return type; }

        // 获取DMA文件描述符（仅适用于DMABUF类型）
        int get_dma_fd() const { return dma_fd; }

        // 获取缓冲区大小
        size_t get_size() const { return size; }

    private:
        Type type;     // 缓冲区类型
        size_t size;   // 缓冲区大小
        uint8_t *data; // 数据指针（普通内存或DMA内存）
        int dma_fd;    // DMA缓冲区的文件描述符
    };

    // Plane 类，继承自 Buffer
    class Plane : public Buffer
    {
    public:
        Plane(Buffer::Type type, size_t size, int dma_fd, size_t width, size_t height, size_t stride)
            : Buffer(type, size, dma_fd), width(width), height(height), stride(stride) {}

        size_t get_width() const { return width; }
        size_t get_height() const { return height; }
        size_t get_stride() const { return stride; }

        // 转换为 OpenCL 缓冲区
        cl_mem to_cl_mem(cl_context context, cl_command_queue queue, bool is_dma = false)
        {
            if (is_dma && get_type() == Buffer::Type::DMABUF)
            {
                // 使用 DMA FD 创建 cl_mem
                int fd = get_dma_fd();
                cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
                return clCreateBufferFromFd(context, fd, flags, get_stride() * get_height(), nullptr);
            }
            else
            {
                // 使用普通内存创建 cl_mem
                uint8_t *ptr = get_data();
                if (ptr == nullptr)
                {
                    std::cerr << "Failed to get plane data." << std::endl;
                    return nullptr;
                }
                cl_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
                return clCreateBuffer(context, flags, get_stride() * get_height(), ptr, nullptr);
            }
        }

    private:
        size_t width, height, stride;  // 图像平面特定的属性
    };

    // BufferPool 类
    class BufferPool
    {
    public:
        BufferPool(const std::vector<size_t> &buffer_sizes)
            : buffer_sizes(buffer_sizes)
        {
            allocate_buffers();
        }

        // 获取缓冲区
        std::shared_ptr<Buffer> get_buffer(Buffer::Type type, size_t size)
        {
            std::lock_guard<std::mutex> lock(mutex);

            // 根据大小选择对应的池
            auto &pool = buffer_pools[size];
            for (auto &buffer : pool)
            {
                if (!buffer_in_use[buffer.get()])
                {
                    buffer_in_use[buffer.get()] = true;
                    return buffer;
                }
            }

            // 如果没有可用的缓存区，创建新的
            auto new_buffer = std::make_shared<Buffer>(type, size);
            pool.push_back(new_buffer);
            buffer_in_use[new_buffer.get()] = true;
            return new_buffer;
        }

        // 归还缓冲区
        void return_buffer(std::shared_ptr<Buffer> buffer)
        {
            std::lock_guard<std::mutex> lock(mutex);
            buffer_in_use[buffer.get()] = false;
        }

        // 缓存和复用 cl_mem
        cl_mem get_cl_mem_from_plane(const Plane &plane, cl_context context, cl_command_queue queue, bool is_dma = false)
        {
            std::lock_guard<std::mutex> lock(cl_mem_mutex);

            // 尝试从缓存中获取
            auto key = std::make_tuple(plane.get_width(), plane.get_height(), plane.get_stride(), is_dma);
            auto it = cl_mem_cache.find(key);
            if (it != cl_mem_cache.end())
            {
                // 缓存命中，直接返回
                return it->second;
            }

            // 创建新的 cl_mem 对象
            cl_mem cl_mem_obj = plane.to_cl_mem(context, queue, is_dma);
            // 缓存该 cl_mem
            cl_mem_cache[key] = cl_mem_obj;
            return cl_mem_obj;
        }

        // 归还 cl_mem 缓存
        void return_cl_mem(const Plane &plane, cl_mem cl_mem_obj)
        {
            std::lock_guard<std::mutex> lock(cl_mem_mutex);
            auto key = std::make_tuple(plane.get_width(), plane.get_height(), plane.get_stride(), false); // 默认是普通内存
            cl_mem_cache.erase(key);
            // OpenCL 没有直接提供销毁缓冲区的 API，但可以调用 release 等方法释放内存
            clReleaseMemObject(cl_mem_obj);
        }

    private:
        std::unordered_map<size_t, std::vector<std::shared_ptr<Buffer>>> buffer_pools; // 存储不同大小的缓冲池
        std::unordered_map<Buffer *, bool> buffer_in_use;                              // 跟踪每个缓冲区是否正在使用
        std::mutex mutex;
        std::vector<size_t> buffer_sizes; // 支持的不同大小的缓冲区
        // cl_mem 缓存
        std::unordered_map<std::tuple<size_t, size_t, size_t, bool>, cl_mem> cl_mem_cache;
        std::mutex cl_mem_mutex;

        void allocate_buffers()
        {
            // 为每个支持的大小分配内存池
            for (auto size : buffer_sizes)
            {
                buffer_pools[size] = std::vector<std::shared_ptr<Buffer>>();
            }
        }
    };
}
