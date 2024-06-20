import os


class DirectoryGenerator:
    """
    A class to generate directories for the audio data.

    Attributes:
    ----------
    None

    Methods:
    -------
    main():
        Creates the necessary directories for the audio data.
    """
    def main():
        # Create directories
        paths = ['./data/ai_full', 
                './data/human_full', 
                './data/ai_converted', 
                './data/human_converted', 
                './data/ai_split', 
                './data/human_split', 
                './data/validation_set/human_full',
                './data/validation_set/human_converted',
                './data/validation_set/human_split', 
                './data/validation_set/ai_full',
                './data/validation_set/ai_converted',
                './data/validation_set/ai_split',
                
                
            ]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

if __name__ == "__main__":
    DirectoryGenerator.main()
