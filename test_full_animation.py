import sys
sys.path.insert(0, 'src')
import pose_sequences
import dance_generator

# Generate a simple choreography
choreo = dance_generator.generate_choreography(
    beats=[1, 2, 3, 4, 5, 6, 7, 8],
    tempo_bpm=120,
    difficulty='easy',
    max_steps=8
)

print(f'Generated {len(choreo)} steps')

# Try to create full animation
try:
    animation = pose_sequences.create_full_routine_animation(choreo, fps=5)
    if animation:
        print(f'Animation generated: {len(animation)} bytes')
    else:
        print('Animation generation returned None')
except Exception as e:
    print(f'Error: {e}')