#ifndef ARA_DOCUMENT_CONTROLLER_H
#define ARA_DOCUMENT_CONTROLLER_H

#include <cstdint>

namespace mamba {

struct alignas(64) ARAConditioningFrame {
    float tempo_bpm;
    float key_sig;
    float chord_one_hot[12];
    float reserved[2];
};

template<typename T> struct Seqlocked;

struct ARADocumentController {
uint32_t version;
uint32_t document_size;
void* document_instance;

bool (*createDocument)(ARADocumentController* controller, const char* file_path);
bool (*loadDocument)(ARADocumentController* controller, const char* file_path);
bool (*saveDocument)(ARADocumentController* controller, const char* file_path);
void (*destroyDocument)(ARADocumentController* controller);
};

class ARAHostDocument {
public:
    ARAHostDocument();
    ~ARAHostDocument();

    void set_tempo(float bpm);
    void set_key_signature(float key);
    void set_chord(const float* chord_one_hot);

    float get_tempo() const;
    float get_key_signature() const;
    void get_chord(float* out_chord) const;

private:
    float tempo_bpm_;
    float key_sig_;
    float chord_[12];
};

void ARA_set_tempo(float bpm);
void ARA_set_key_signature(float key);
void ARA_set_chord(const float* chord_one_hot);
bool ARA_get_conditioning(float& tempo, float& key, float* chord);
Seqlocked<ARAConditioningFrame>* ARA_get_conditioning_seqlock();

void ARA_write_conditioning_to(Seqlocked<ARAConditioningFrame>* target, float bpm, float key, const float* chord_one_hot);

} // namespace mamba

#endif