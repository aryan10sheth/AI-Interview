import whisper
import tempfile
import os
from textblob import TextBlob
from typing import Dict, List, Optional, Union
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from werkzeug.datastructures import FileStorage

class AudioAnalyzer:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.filler_words = set(['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right'])
        
    def analyze_audio(self, audio_path: str) -> Dict:
        """Analyze audio file and return metrics including transcription."""
        try:
            # Get transcription
            result = self.model.transcribe(audio_path)
            transcription = result["text"]
            
            # Calculate basic metrics
            words = word_tokenize(transcription.lower())
            sentences = sent_tokenize(transcription)
            
            metrics = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
                'unique_words': len(set(words)),
                'filler_word_count': sum(1 for word in words if word in self.filler_words)
            }
            
            return {
                'transcription': transcription,
                'metrics': metrics,
                'sentiment': self._analyze_sentiment(transcription)
            }
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return None
            
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of transcribed text."""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Convert polarity to label
            if sentiment.polarity > 0.5:
                label = 'very_positive'
            elif sentiment.polarity > 0:
                label = 'positive'
            elif sentiment.polarity < -0.5:
                label = 'very_negative'
            elif sentiment.polarity < 0:
                label = 'negative'
            else:
                label = 'neutral'
                
            return {
                'polarity': float(sentiment.polarity),
                'subjectivity': float(sentiment.subjectivity),
                'sentiment_label': label
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'sentiment_label': 'neutral'
            }
        
    def _calculate_speech_metrics(self, words: List[str], sentences: List[str]) -> Dict:
        """Calculate various speech metrics."""
        num_words = len(words)
        num_sentences = len(sentences)
        
        # Calculate filler word usage
        filler_count = sum(1 for word in words if word in self.filler_words)
        filler_ratio = filler_count / num_words if num_words > 0 else 0
        
        # Calculate average sentence length
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        
        # Calculate speech complexity
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / num_words if num_words > 0 else 0
        
        return {
            'word_count': num_words,
            'sentence_count': num_sentences,
            'filler_word_count': filler_count,
            'filler_word_ratio': filler_ratio,
            'avg_sentence_length': avg_sentence_length,
            'vocabulary_diversity': vocabulary_diversity
        }
        
    def _analyze_confidence(self, words: List[str], sentences: List[str]) -> float:
        """Analyze speech confidence based on various factors."""
        # Confidence indicators
        confidence_markers = set(['believe', 'think', 'know', 'sure', 'confident', 'definitely'])
        uncertainty_markers = set(['maybe', 'perhaps', 'possibly', 'might', 'guess', 'sort of', 'kind of'])
        
        # Count markers
        confidence_count = sum(1 for word in words if word in confidence_markers)
        uncertainty_count = sum(1 for word in words if word in uncertainty_markers)
        
        # Calculate base confidence score
        total_markers = confidence_count + uncertainty_count
        if total_markers == 0:
            base_confidence = 0.5
        else:
            base_confidence = confidence_count / total_markers
            
        return base_confidence
        
    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert sentiment polarity to human-readable label."""
        if polarity > 0.3:
            return 'very_positive'
        elif polarity > 0:
            return 'positive'
        elif polarity < -0.3:
            return 'very_negative'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral' 