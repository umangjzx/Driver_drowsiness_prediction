"""Main entrypoint that forwards to the enhanced Hugging Face detector."""

from realtime_detector_enhanced import EnhancedDrowsinessDetector


def main():
    detector = EnhancedDrowsinessDetector(use_huggingface=True, hf_backend="local")
    detector.run()


if __name__ == "__main__":
    main()
