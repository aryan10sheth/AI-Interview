import numpy as np
from textblob import TextBlob
from typing import Dict, List
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import spacy

class ResponseAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            
        self.nlp = spacy.load("en_core_web_md")  # Use medium model with word vectors
        self.stop_words = set(stopwords.words('english'))
        self.filler_words = set(['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right'])
        
    def analyze_response(self, text: str, question: str) -> Dict:
        """Analyze interview response and return comprehensive analysis."""
        try:
            # Basic text analysis
            words = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
            
            # Calculate metrics
            metrics = self._calculate_response_metrics(words, sentences, question)
            
            # Content analysis
            content_scores = self._analyze_content(text, question)
            
            # Sentiment analysis
            sentiment_scores = self._analyze_sentiment(text)
            
            # Confidence analysis
            confidence_score = self._analyze_confidence(words, sentences)
            
            return {
                'transcription': text,
                'metrics': metrics,
                'content': content_scores,
                'sentiment': sentiment_scores,
                'confidence_score': confidence_score,
                'overall_score': self._calculate_overall_score(metrics, content_scores, sentiment_scores, confidence_score)
            }
            
        except Exception as e:
            print(f"Error in response analysis: {str(e)}")
            return None
            
    def _calculate_response_metrics(self, words: List[str], sentences: List[str], question: str) -> Dict:
        """Calculate basic speech metrics."""
        # Remove stop words and filler words
        content_words = [w for w in words if w not in self.stop_words and w not in self.filler_words]
        
        metrics = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(content_words)),
            'filler_word_count': sum(1 for word in words if word in self.filler_words),
            'vocabulary_richness': len(set(content_words)) / len(content_words) if content_words else 0
        }
        
        return metrics
        
    def _analyze_content(self, text: str, question: str) -> Dict:
        """Analyze the content quality and relevance."""
        doc = self.nlp(text)
        question_doc = self.nlp(question)
        
        # Extract key entities and noun phrases
        entities = [ent.text for ent in doc.ents]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Calculate relevance score based on semantic similarity
        relevance_score = doc.similarity(question_doc)
        
        # Analyze structure and coherence
        sentence_transitions = self._analyze_transitions(doc)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness(doc, question_doc)
        
        return {
            'relevance_score': float(relevance_score),
            'completeness_score': float(completeness_score),
            'key_entities': entities[:5],  # Top 5 entities
            'key_phrases': noun_phrases[:5],  # Top 5 phrases
            'structure_score': sentence_transitions,
            'is_comprehensive': completeness_score > 0.7
        }
        
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment and emotional tone."""
        blob = TextBlob(text)
        
        return {
            'polarity': float(blob.sentiment.polarity),
            'subjectivity': float(blob.sentiment.subjectivity),
            'tone': self._determine_tone(blob.sentiment.polarity)
        }
        
    def _analyze_confidence(self, words: List[str], sentences: List[str]) -> float:
        """Analyze confidence level in speech."""
        # Confidence indicators
        confidence_words = set(['confident', 'sure', 'certain', 'definitely', 'absolutely'])
        uncertainty_words = set(['maybe', 'perhaps', 'guess', 'think', 'possibly'])
        
        # Count indicators
        confident_count = sum(1 for word in words if word in confidence_words)
        uncertainty_count = sum(1 for word in words if word in uncertainty_words)
        
        # Calculate base confidence score
        total_indicators = confident_count + uncertainty_count
        if total_indicators == 0:
            base_score = 0.5
        else:
            base_score = confident_count / total_indicators
            
        # Adjust score based on sentence structure
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        length_modifier = min(1.0, avg_sentence_length / 20)  # Optimal length around 15-20 words
        
        return min(1.0, base_score * 0.7 + length_modifier * 0.3)
        
    def _analyze_transitions(self, doc) -> float:
        """Analyze sentence transitions and coherence."""
        transition_words = set(['however', 'therefore', 'furthermore', 'moreover', 'consequently'])
        sentence_count = len(list(doc.sents))
        
        if sentence_count <= 1:
            return 0.5
            
        transition_count = sum(1 for token in doc if token.text.lower() in transition_words)
        return min(1.0, transition_count / (sentence_count - 1))
        
    def _calculate_completeness(self, response_doc, question_doc) -> float:
        """Calculate how completely the response addresses the question."""
        # Extract key components from question
        question_keywords = [token.text for token in question_doc if not token.is_stop and token.is_alpha]
        
        # Check coverage in response
        response_text = response_doc.text.lower()
        covered_keywords = sum(1 for keyword in question_keywords if keyword.lower() in response_text)
        
        # Base completeness score
        base_score = covered_keywords / len(question_keywords) if question_keywords else 0.5
        
        # Adjust for response length and detail
        length_score = min(1.0, len(response_doc) / 100)  # Normalize for responses around 100 tokens
        
        return (base_score * 0.7 + length_score * 0.3)
        
    def _determine_tone(self, polarity: float) -> str:
        """Determine the overall tone based on polarity."""
        if polarity >= 0.5:
            return 'very positive'
        elif polarity > 0:
            return 'positive'
        elif polarity == 0:
            return 'neutral'
        elif polarity > -0.5:
            return 'negative'
        else:
            return 'very negative'
            
    def _calculate_overall_score(self, metrics: Dict, content: Dict, sentiment: Dict, confidence: float) -> float:
        """Calculate overall response quality score."""
        # Weights for different components
        weights = {
            'content_relevance': 0.3,
            'completeness': 0.2,
            'confidence': 0.2,
            'structure': 0.15,
            'vocabulary': 0.15
        }
        
        # Calculate component scores
        scores = {
            'content_relevance': content['relevance_score'],
            'completeness': content['completeness_score'],
            'confidence': confidence,
            'structure': content['structure_score'],
            'vocabulary': metrics['vocabulary_richness']
        }
        
        # Calculate weighted average
        overall_score = sum(score * weights[component] for component, score in scores.items())
        
        return min(1.0, max(0.0, overall_score)) 