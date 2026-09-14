#include "core/arena_allocator.h"
#include <cstdlib>
#include <cstring>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace emt {

ArenaAllocator::ArenaAllocator(size_t size) : size_(size), offset_(0), is_mmaped_(false) {
#if defined(__unix__) || defined(__APPLE__)
    memory_ = static_cast<uint8_t*>(mmap(nullptr, size_, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (memory_ != MAP_FAILED) {
        is_mmaped_ = true;
    } else {
        memory_ = static_cast<uint8_t*>(std::malloc(size_));
    }
#elif defined(_WIN32)
    memory_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, size_, MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE));
    if (memory_) {
        is_mmaped_ = true;
    } else {
        memory_ = static_cast<uint8_t*>(std::malloc(size_));
    }
#else
    memory_ = static_cast<uint8_t*>(std::malloc(size_));
#endif
}

ArenaAllocator::~ArenaAllocator() {
#if defined(__unix__) || defined(__APPLE__)
    if (is_mmaped_) {
        munmap(memory_, size_);
    } else {
        std::free(memory_);
    }
#elif defined(_WIN32)
    if (is_mmaped_) {
        VirtualFree(memory_, 0, MEM_RELEASE);
    } else {
        std::free(memory_);
    }
#else
    std::free(memory_);
#endif
}

void* ArenaAllocator::allocate(size_t size, size_t alignment) {
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned_offset + size > size_) {
        return nullptr;
    }
    void* ptr = memory_ + aligned_offset;
    offset_ = aligned_offset + size;
    return ptr;
}

void ArenaAllocator::reset() {
    offset_ = 0;
    std::memset(memory_, 0, size_);
}

}