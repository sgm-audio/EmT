#ifndef ARENA_ALLOCATOR_H
#define ARENA_ALLOCATOR_H

#include <cstddef>
#include <cstdint>

namespace mamba {

class ArenaAllocator {
public:
    ArenaAllocator(size_t size);
    ~ArenaAllocator();

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    void* allocate(size_t size, size_t alignment = 16);

    void reset();

    size_t used() const { return offset_; }
    size_t capacity() const { return size_; }

private:
    uint8_t* memory_;
    size_t size_;
    size_t offset_;
    bool is_mmaped_;
};

template<typename T, size_t Alignment = 64>
class StlAllocator {
public:
    using value_type = T;

    StlAllocator() = default;
    template<typename U> StlAllocator(const StlAllocator<U, Alignment>&) {}

    T* allocate(size_t n) {
        void* ptr = arena_->allocate(n * sizeof(T), Alignment);
        return static_cast<T*>(ptr);
    }

    void deallocate(T*, size_t) {}

    void set_arena(ArenaAllocator* arena) { arena_ = arena; }

private:
    template<typename U, size_t A> friend struct StlAllocator;
    ArenaAllocator* arena_ = nullptr;
};

} // namespace mamba

#endif