class TextComparator:
    def __init__(self, actual_text):
        self.original_actual = actual_text
        self.actual_words = actual_text.split()
        self.last_complete_words = []  # To track complete words in prediction
        self.current_partial = ""  # To track the current incomplete word
        self.last_word = False
    
    def normalize_word(self, word):
        """Normalize word by removing diacritics only"""
        return ''.join(c for c in word if not (0x064B <= ord(c) <= 0x065F))
    
    def process_partial(self, partial_text):
        """Process partial text and return True if a new complete word is detected"""
        words = partial_text.split(' ')
        
        # If we have more segments than before, a new word was completed
        if len(words) > len(self.last_complete_words) + 1: 
            # All words except last are complete  
            new_complete_words = words[:-1]
            self.last_complete_words = new_complete_words
            self.current_partial = words[-1]
            return True
        else:
            # Just update the current partial word
            if words:
                self.current_partial = words[-1]
            return False
    
    def get_current_prediction(self):
        """Get the full current prediction (complete words + current partial)"""
        return ' '.join(self.last_complete_words + [self.current_partial]) if self.current_partial else ' '.join(self.last_complete_words)
    
    def similarity(self, a, b):
        """Check the similarity ratio between words and a & b"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()

    def compare_latest_word(self):
        """Compare the last complete word with actual text"""
        if not self.last_complete_words:
            return []
        
        # Get the index of the last complete word
        last_word_index = len(self.last_complete_words) - 1

        # Check if all words compared
        if len(self.last_complete_words) > len(self.actual_words):
            self.last_word = False
            return []
        
        actual_word = self.actual_words[last_word_index]
        predicted_word = self.last_complete_words[last_word_index]
        
        # Similarity check
        sim = self.similarity(self.normalize_word(predicted_word), self.normalize_word(actual_word))

        # If its seconds last word, flag for upcoming last word as TRUE 
        if len(self.last_complete_words) == len(self.actual_words) - 1:
            self.last_word = True

        if sim < 0.75:
            return [actual_word]
        return []