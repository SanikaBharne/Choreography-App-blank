import source_separation
import beat_detection


def main():
    file_path = input("Insert your file path: ")
    stems = source_separation.separate(file_path)
    drum_beats = beat_detection.detect_beats(stems["drums"])
    mix_beats = beat_detection.detect_beats(file_path)
    merged_beats = beat_detection.merge_beats(drum_beats, mix_beats)
    print(merged_beats)
    source_separation.get_instrumental(stems, output_path="instrumental.wav")

if __name__ == "__main__":
    main()
