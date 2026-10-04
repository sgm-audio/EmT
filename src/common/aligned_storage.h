#ifndef EMT_COMMON_ALIGNED_STORAGE_H
#define EMT_COMMON_ALIGNED_STORAGE_H

#include <cstddef>
#include <new>

namespace mamba {

// Backing storage for objects with manually-managed lifetime (placement new)
// held as a member of a larger struct.
//
// Why this exists
// ---------------
// A plain `std::byte storage[sizeof(T)];` member inherits only the alignment of
// the *member*, not of the enclosing struct.  If the member follows a pointer
// or another smaller member, the compiler is free to place it at an offset that
// does not satisfy `alignof(T)`.  Running a constructor for an over-aligned T
// (for example `SSMInferenceCore`, which is `alignas(64)`) at such an address is
// undefined behaviour, and on x86-64 it faults as soon as the compiler emits an
// aligned SIMD store.
//
// `AlignedStorage` carries the alignment requirement on the member itself, so
// the guarantee is structural rather than a comment that can silently drift:
//
//     struct Host {
//         const clap_host_t* host;              // may be followed by padding
//         AlignedStorage<SSMInferenceCore, 64> core_storage;
//     };
//
//     auto* core = new (core_storage.data) SSMInferenceCore(cfg);
//
// `Alignment` defaults to `alignof(T)`; pass a larger value when the whole
// object (past the first member) must stay cache-line aligned.
template <typename T, std::size_t Alignment = alignof(T)>
struct alignas(Alignment) AlignedStorage {
    static_assert(Alignment >= alignof(T),
                  "AlignedStorage alignment must be at least alignof(T)");
    static_assert((Alignment & (Alignment - 1)) == 0,
                  "AlignedStorage alignment must be a power of two");
    static_assert(sizeof(T) > 0, "AlignedStorage requires a complete type");

    std::byte data[sizeof(T)];

    T*       ptr()       { return reinterpret_cast<T*>(data); }
    const T* ptr() const { return reinterpret_cast<const T*>(data); }
};

} // namespace mamba

#endif // EMT_COMMON_ALIGNED_STORAGE_H
