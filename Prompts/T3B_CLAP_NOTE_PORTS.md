# Worker Task Prompt (T3b) - CLAP Note Ports Implementation

## ROLE
CLAP Integration Engineer implementing note event handling for the Mamba Synth plugin.

## CONTEXT
This task focuses on implementing the CLAP note-ports extension for the Mamba Synth plugin, enabling it to receive MIDI note events from the host and translate them into conditioning signals for the SSM inference core. This is a critical first step in connecting the audio processing core to the host DAW via CLAP.

## INSTRUCTIONS
1. Implement the CLAP note-ports extension in the Mamba Synth plugin
2. Handle note-on and note-off events from the host via the CLAP note-ports interface
3. Translate incoming MIDI notes (note number, velocity) into conditioning signals that modulate SSM parameters (Δ, B, C) via learned projections
4. Implement voice management for polyphonic note handling (optional for first pass)
5. Ensure all note processing in the audio thread is lock-free and allocation-free
6. Use the thread-safe communication mechanisms defined in the Integration Contract to send note events to the conditioning system
7. Implement basic unit tests to verify note event handling
8. Test with a CLAP host (e.g., Carla, Bitwig Studio in CLAP mode) to verify note reception

## CONSTRAINTS
- **Audio Thread Safety**: Note event processing must contain no heap allocations or mutex locks
- **CLAP Compliance**: Must follow CLAP note-ports extension specification
- **Memory**: Must use pre-allocated buffers for any note event storage
- **Real-time**: Note processing must complete within audio buffer deadline
- **Conditioning Interface**: Must interface cleanly with the SSM core's parameter modulation system
- **Dependencies**: Only CLAP header and C++ standard library (plus SSM core)
- **Testing**: Must be testable with automated unit tests and manual host testing

## OUTPUT FORMAT
1. Modified plugin processor file showing:
   - Implementation of CLAP note-ports extension methods (`note_on`, `note_off`)
   - Translation of MIDI note events to conditioning signals
   - Thread-safe communication with conditioning system (lock-free queue or atomic updates)
2. Conditioning system interface that:
   - Accepts note-based conditioning requests from audio thread
   - Maps note number/velocity to SSM parameter modifications (Δ, B, C)
   - Uses learned projections (to be implemented later, stub for now)
3. Unit test file (`test_clap_note_ports.cpp`) that:
   - Simulates note-on/note-off events
   - Verifies proper translation to conditioning signals
   - Tests thread safety of the interface
4. Basic integration test instructions for manual verification in a CLAP host

## ACCEPTANCE CRITERIA
- [ ] CLAP note-on and note-off methods contain no heap allocations or mutex locks
- [ ] Note events are properly translated to conditioning signals (stub mapping acceptable for now)
- [ ] Communication with conditioning system is thread-safe and lock-free
- [ ] Unit tests pass and verify note event handling
- [ ] Plugin loads correctly in a CLAP host and receives MIDI note events
- [ ] Clear separation between CLAP interface and audio processing core
- [ ] Documentation on how note events map to SSM parameter modulation