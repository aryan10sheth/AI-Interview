from typing import Dict, List, Optional, Any
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
import random
import nltk
from textblob import TextBlob
import json
import spacy
from utils.pdf_processor import extract_text_from_pdf

class RecruiterEvaluator:
    def __init__(self):
        self.mistral_client = MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))
        self.responses = []
        self.questions = []
        self.transcriptions = []  # Store all transcriptions
        self.nlp = spacy.load('en_core_web_md')
        self.model = "mistral-medium"
        
    def generate_interview_questions(self, pdf_path: str, job_description: str) -> List[str]:
        """
        Generate 3 resume-based questions using the LLM and 2 random personal questions.
        pdf_path: Path to the uploaded PDF resume.
        job_description: The job description text.
        """
        # Extract text from the uploaded PDF
        cv_text = extract_text_from_pdf(pdf_path)
        if not cv_text:
            print("Failed to extract text from PDF, using fallback questions.")
            return self._get_fallback_questions()

        # Analyze the CV and generate 3 technical questions
        try:
            question_gen_messages = [
                ChatMessage(role="system", content="""You are an expert technical recruiter. Based on the candidate's resume and the job description, generate 3 specific technical interview questions that:
- Are directly related to the candidate's experience
- Allow them to demonstrate their expertise
- Focus on their most significant projects or achievements
- Validate their claimed skills
- Are open-ended and encourage detailed responses
Format: Return only the 3 questions, one per line, without numbering."""),
                ChatMessage(role="user", content=f"Resume Text:\n{cv_text}\n\nJob Description:\n{job_description}")
            ]
            tech_questions_response = self.mistral_client.chat(
                model=self.model,
                messages=question_gen_messages
            )
            tech_questions = [q.strip() for q in tech_questions_response.choices[0].message.content.split('\n') if q.strip()][:3]
        except Exception as e:
            print(f"Error generating technical questions: {e}")
            tech_questions = self._get_fallback_questions()[:3]

        # Pool of personal questions
        personal_questions_pool = [
            "What motivates you to excel in your work?",
            "How do you handle challenging situations in a team environment?",
            "What are your career goals and how does this position align with them?",
            "How do you stay updated with the latest industry trends?",
            "Describe a situation where you had to learn a new technology quickly.",
            "How do you approach problem-solving in your work?",
            "What's your ideal work environment?",
            "How do you handle feedback and criticism?",
            "What interests you most about this position?",
            "How do you manage work-life balance?"
        ]
        # Randomly select 2 personal questions
        personal_questions = random.sample(personal_questions_pool, 2)
        
        all_questions = tech_questions + personal_questions
        self.questions = all_questions
        return all_questions
    
    def store_response(self, question: str, transcription: str, evaluation: Dict):
        """Store response data for final evaluation."""
        self.transcriptions.append({
            'question': question,
            'transcription': transcription,
            'evaluation': evaluation
        })

    def generate_final_report(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive final report based on all responses.
        Provides a clear hiring recommendation and overall score.
        """
        if len(responses) < 5:
            raise ValueError(f"Not enough responses. Expected 5, got {len(responses)}")
        
        # Prepare the context for the LLM
        responses_context = "\n\n".join([
            f"Question {i+1}: {r['question']}\nAnswer: {r['transcription']}\nIndividual Score: {r['evaluation']['score']}/10\nFeedback: {r['evaluation']['feedback']}"
            for i, r in enumerate(responses)
        ])
        
        messages = [
            ChatMessage(role="system", content="""You are an expert technical recruiter making a final hiring decision.
            Review all interview responses and provide:
            1. A clear HIRE/NO HIRE recommendation
            2. Final score out of 10
            3. Brief explanation of your decision
            4. Key strengths across all responses
            5. Critical areas for improvement
            
            Format as JSON with these exact keys:
            {
                "recommendation": "HIRE" or "NO HIRE",
                "final_score": number,
                "decision_explanation": "clear explanation",
                "overall_strengths": ["point1", "point2"],
                "overall_improvements": ["point1", "point2"]
            }"""),
            ChatMessage(role="user", content=f"Complete Interview Responses:\n{responses_context}\n\nProvide your final evaluation following the format above.")
        ]
        
        response = self.mistral_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        try:
            report = json.loads(response.choices[0].message.content)
            # Ensure all required fields exist
            report.setdefault('recommendation', 'NO HIRE')
            report.setdefault('final_score', 5)
            report.setdefault('decision_explanation', 'Insufficient data for complete evaluation')
            report.setdefault('overall_strengths', ['Completed the interview process'])
            report.setdefault('overall_improvements', ['Need more detailed responses'])
            
        except Exception as e:
            print(f"Error generating final report: {str(e)}")
            report = {
                'recommendation': 'NO HIRE',
                'final_score': 5,
                'decision_explanation': 'Error generating final evaluation',
                'overall_strengths': ['Completed the interview process'],
                'overall_improvements': ['Technical evaluation incomplete']
            }
        
        return report

    def evaluate(self, question: str, transcription: str, video_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate the candidate's response using LLM.
        Focuses on answer quality and relevance.
        """
        messages = [
            ChatMessage(role="system", content="""You are an expert technical recruiter evaluating interview responses.
            Analyze the candidate's answer and provide:
            1. A score out of 10 based on:
               - Relevance to the question
               - Technical accuracy
               - Clarity of explanation
               - Depth of knowledge
            2. Brief but specific feedback on the answer
            3. Key strengths (2-3 points)
            4. Areas for improvement (1-2 points)
            
            Format your response as JSON with these exact keys:
            {
                "score": number,
                "feedback": "clear explanation of score",
                "strengths": ["point1", "point2"],
                "improvements": ["point1", "point2"]
            }"""),
            ChatMessage(role="user", content=f"""Question: {question}
            
            Candidate's Answer: {transcription}
            
            Evaluate this response following the format above.""")
        ]
        
        response = self.mistral_client.chat(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        try:
            evaluation = json.loads(response.choices[0].message.content)
            # Ensure all required fields exist with proper types
            evaluation['score'] = float(evaluation.get('score', 5))
            evaluation['feedback'] = str(evaluation.get('feedback', 'No feedback available'))
            evaluation['strengths'] = list(evaluation.get('strengths', ['Response provided']))
            evaluation['improvements'] = list(evaluation.get('improvements', ['Consider providing more detail']))
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            evaluation = {
                "score": 5,
                "feedback": "Unable to generate detailed feedback. Please review the response manually.",
                "strengths": ["Response provided"],
                "improvements": ["Consider providing more detail"]
            }
        
        # Store response for final evaluation
        self.responses.append({
            'question': question,
            'transcription': transcription,
            'evaluation': evaluation
        })
        
        return evaluation

    def _calculate_relevance(self, answer_text: str) -> float:
        """Calculate relevance score using keyword matching."""
        if not self.questions:
            return 0.7
            
        # Get keywords from the current question
        current_question = self.questions[len(self.responses)]
        question_words = set(nltk.word_tokenize(current_question.lower()))
        answer_words = set(nltk.word_tokenize(answer_text.lower()))
        
        # Calculate overlap
        common_words = question_words.intersection(answer_words)
        relevance_score = len(common_words) / len(question_words) if question_words else 0.7
        
        return min(1.0, relevance_score + 0.3)  # Add base relevance
    
    def _get_tone_label(self, polarity: float) -> str:
        """Convert sentiment polarity to tone label."""
        if polarity > 0.5:
            return 'very positive'
        elif polarity > 0:
            return 'positive'
        elif polarity < -0.5:
            return 'very negative'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral'
            
    def generate_report(self) -> str:
        """Generate a JSON report of the interview analysis."""
        try:
            report_data = {
                'interview_summary': {
                    'total_questions': len(self.questions),
                    'total_responses': len(self.responses),
                    'overall_performance': self._calculate_overall_performance()
                },
                'detailed_responses': [{
                    'question': self.questions[i],
                    'response': resp['answer'],
                    'evaluation': resp['evaluation']
                } for i, resp in enumerate(self.responses)]
            }
            
            return json.dumps(report_data, indent=2)
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return json.dumps({'error': str(e)})
            
    def _calculate_overall_performance(self) -> Dict:
        """Calculate overall interview performance metrics."""
        if not self.responses:
            return {
                'score': 0,
                'strengths': [],
                'areas_for_improvement': []
            }
            
        total_score = 0
        strengths = []
        areas_for_improvement = []
        
        for i, response in enumerate(self.responses):
            eval_data = response['evaluation']
            question = self.questions[i]
            answer = response['answer']
            
            # Calculate question score (out of 10)
            question_score = 0
            
            # Content analysis (40% of score)
            if 'content_analysis' in eval_data:
                content = eval_data['content_analysis']
                question_score += content['relevance_score'] * 2  # 20%
                question_score += content['clarity_score'] * 2    # 20%
                
                if content['word_count'] < 50:
                    areas_for_improvement.append("Provide more detailed responses")
                elif content['word_count'] > 300:
                    areas_for_improvement.append("Try to be more concise in your responses")
                    
                if content['vocabulary_richness'] > 0.7:
                    strengths.append("Strong vocabulary and articulate expression")
            
            # Video analysis (40% of score)
            if 'video_analysis' in eval_data:
                video = eval_data['video_analysis']
                question_score += video['eye_contact_score'] * 2     # 20%
                question_score += video['confidence_score'] * 2      # 20%
                
                if video['eye_contact_score'] < 0.6:
                    areas_for_improvement.append("Maintain better eye contact")
                elif video['eye_contact_score'] > 0.8:
                    strengths.append("Excellent eye contact and engagement")
                    
                if video['confidence_score'] < 0.6:
                    areas_for_improvement.append("Work on projecting more confidence")
                elif video['confidence_score'] > 0.8:
                    strengths.append("Strong confidence and presence")
            
            # Sentiment analysis (20% of score)
            if 'sentiment_analysis' in eval_data:
                sentiment = eval_data['sentiment_analysis']
                sentiment_score = (1 + sentiment['polarity']) / 2  # Convert -1:1 to 0:1
                question_score += sentiment_score * 2  # 20%
                
                if sentiment['polarity'] > 0.5:
                    strengths.append("Positive and enthusiastic attitude")
                elif sentiment['polarity'] < -0.2:
                    areas_for_improvement.append("Maintain a more positive tone")
            
            total_score += question_score
        
        # Calculate average score
        avg_score = total_score / len(self.responses)
        
        # Remove duplicates and limit to top 3
        strengths = list(set(strengths))[:3]
        areas_for_improvement = list(set(areas_for_improvement))[:3]
        
        return {
            'score': round(avg_score, 1),  # Score out of 10
            'strengths': strengths,
            'areas_for_improvement': areas_for_improvement
        }
    
    def _evaluate_content(self, response_analysis):
        """Evaluate the content quality of the response."""
        if not response_analysis:
            return 0.5
            
        metrics = {
            'relevance': response_analysis.get('content', {}).get('relevance_score', 0.5),
            'completeness': response_analysis.get('content', {}).get('completeness_score', 0.5),
            'structure': response_analysis.get('content', {}).get('structure_score', 0.5)
        }
        
        return sum(metrics.values()) / len(metrics)

    def _evaluate_delivery(self, video_analysis):
        """Evaluate the delivery aspects from video analysis."""
        if not video_analysis:
            return 0.5
            
        metrics = {
            'eye_contact': video_analysis.get('eye_contact_score', 0.5),
            'engagement': video_analysis.get('engagement_score', 0.5),
            'confidence': video_analysis.get('confidence_score', 0.5)
        }
        
        return sum(metrics.values()) / len(metrics)

    def _evaluate_communication(self, response_analysis):
        """Evaluate communication effectiveness."""
        if not response_analysis:
            return 0.5
            
        metrics = response_analysis.get('metrics', {})
        
        # Calculate vocabulary score
        vocab_richness = metrics.get('vocabulary_richness', 0.5)
        vocab_score = min(1.0, vocab_richness * 1.5)  # Scale up but cap at 1.0
        
        # Calculate fluency score
        words_per_sentence = metrics.get('avg_words_per_sentence', 15)
        fluency_score = min(1.0, words_per_sentence / 20)  # Optimal around 15-20 words
        
        # Calculate filler word penalty
        word_count = metrics.get('word_count', 100)
        filler_count = metrics.get('filler_word_count', 0)
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        filler_penalty = max(0, 1 - (filler_ratio * 10))  # Penalize heavy filler usage
        
        return (vocab_score + fluency_score + filler_penalty) / 3

    def _get_fallback_questions(self) -> List[str]:
        """Return fallback questions if question generation fails."""
        return [
            "What inspired you to pursue a career in this field?",
            "How do you handle challenging technical problems?",
            "Can you explain your experience with the main technologies mentioned in your resume?",
            "What kind of work environment helps you thrive?",
            "How would you approach solving a complex technical problem in this role?"
        ]
    
    def _get_fallback_evaluation(self) -> Dict:
        """Return fallback evaluation if evaluation fails."""
        return {
            'final_score': 5.0,
            'recommendation': 'Consider with Reservations',
            'behavioral_assessment': {
                'score': 5.0,
                'attention_level': 'medium',
                'eye_contact': 0.5,
                'body_language': 'Unable to analyze body language'
            },
            'communication_assessment': {
                'score': 5.0,
                'clarity': 5.0,
                'confidence': 0.5,
                'filler_words': 0
            },
            'content_assessment': {
                'content_score': 5.0,
                'strengths': ['Unable to analyze strengths'],
                'improvements': ['Unable to analyze improvements']
            },
            'detailed_feedback': {
                'strengths': ['Unable to analyze strengths'],
                'areas_for_improvement': ['Unable to analyze improvements'],
                'interview_presence': 'Unable to evaluate interview presence'
            }
        }
    
    def _calculate_behavioral_score(self, video_analysis: Dict) -> float:
        """Calculate score based on behavioral indicators."""
        attention_weights = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4
        }
        
        attention_score = attention_weights.get(video_analysis.get('engagement_level', 'medium'), 0.7)
        eye_contact_score = video_analysis.get('eye_contact_score', 0.5)
        
        return (attention_score * 0.6 + eye_contact_score * 0.4) * 10
    
    def _calculate_communication_score(self, audio_analysis: Dict) -> float:
        """Calculate score based on communication metrics."""
        fluency_score = audio_analysis.get('fluency_score', 0.5)
        filler_ratio = min(1.0, audio_analysis.get('filler_words', 0) / 20)  # Normalize filler words
        confidence_score = audio_analysis.get('confidence_score', 0.5)
        
        return (fluency_score * 0.4 + (1 - filler_ratio) * 0.3 + confidence_score * 0.3) * 10
    
    def _calculate_final_score(self,
                             behavioral_score: float,
                             communication_score: float,
                             content_score: float) -> float:
        """Calculate final candidate score."""
        weights = {
            'behavioral': 0.3,
            'communication': 0.3,
            'content': 0.4
        }
        
        final_score = (
            behavioral_score * weights['behavioral'] +
            communication_score * weights['communication'] +
            content_score * weights['content']
        )
        
        return round(final_score, 1)
    
    def _generate_recommendation(self, final_score: float) -> str:
        """Generate hiring recommendation based on final score."""
        if final_score >= 8.5:
            return 'Strongly Recommend'
        elif final_score >= 7.0:
            return 'Recommend'
        elif final_score >= 5.5:
            return 'Consider with Reservations'
        else:
            return 'Do Not Recommend'
    
    def _interpret_movement_patterns(self, patterns: Dict) -> str:
        """Interpret movement patterns into recruiter-friendly feedback."""
        if not patterns or patterns == 'insufficient_data':
            return 'Insufficient video data for movement analysis'
            
        stable_percentage = patterns.get('stable', 0)
        if stable_percentage > 70:
            return 'Excellent composure and attentiveness'
        elif stable_percentage > 50:
            return 'Good overall composure with some movement'
        elif patterns.get('nodding', 0) > 30:
            return 'Shows active listening through nodding, but could be more stable'
        elif patterns.get('side_to_side', 0) > 30:
            return 'Appears somewhat nervous or uncertain'
        else:
            return 'Significant movement detected, may indicate nervousness'
            
    def _generate_detailed_feedback(self,
                                  behavioral_score: float,
                                  communication_score: float,
                                  content_evaluation: Dict,
                                  video_analysis: Dict,
                                  audio_analysis: Dict) -> Dict:
        """Generate detailed feedback for the recruiter."""
        return {
            'strengths': [
                *content_evaluation['strengths'],
                self._get_behavioral_strength(behavioral_score, video_analysis),
                self._get_communication_strength(communication_score, audio_analysis)
            ],
            'areas_for_improvement': [
                *content_evaluation['improvements'],
                self._get_behavioral_improvement(behavioral_score, video_analysis),
                self._get_communication_improvement(communication_score, audio_analysis)
            ],
            'interview_presence': self._evaluate_interview_presence(video_analysis, audio_analysis)
        }
        
    def _get_behavioral_strength(self, score: float, analysis: Dict) -> str:
        """Generate behavioral strength feedback."""
        if score >= 8:
            return 'Excellent professional demeanor and engagement throughout the interview'
        elif score >= 6:
            return 'Good overall presence with consistent eye contact'
        else:
            return 'Shows potential for improvement in professional presence'
            
    def _get_behavioral_improvement(self, score: float, analysis: Dict) -> str:
        """Generate behavioral improvement feedback."""
        if analysis['movement_patterns'].get('side_to_side', 0) > 30:
            return 'Could improve stability and reduce side-to-side movement'
        elif analysis.get('eye_contact', 0) < 0.6:
            return 'Could maintain more consistent eye contact'
        else:
            return 'Continue working on professional body language'
            
    def _get_communication_strength(self, score: float, analysis: Dict) -> str:
        """Generate communication strength feedback."""
        if score >= 8:
            return 'Excellent verbal communication with clear and confident responses'
        elif score >= 6:
            return 'Good communication skills with room for improvement'
        else:
            return 'Basic communication skills demonstrated'
            
    def _get_communication_improvement(self, score: float, analysis: Dict) -> str:
        """Generate communication improvement feedback."""
        metrics = analysis['metrics']
        if metrics['filler_word_ratio'] > 0.1:
            return 'Reduce use of filler words to improve clarity'
        elif metrics['vocabulary_diversity'] < 0.3:
            return 'Could use more varied vocabulary in responses'
        else:
            return 'Continue working on communication clarity'
            
    def _evaluate_interview_presence(self, video_analysis: Dict, audio_analysis: Dict) -> str:
        """Evaluate overall interview presence."""
        engagement = video_analysis['engagement_level']
        sentiment = audio_analysis['sentiment']['sentiment_label']
        
        if engagement == 'high' and sentiment in ['positive', 'very_positive']:
            return 'Strong and positive interview presence'
        elif engagement == 'medium' and sentiment != 'negative':
            return 'Satisfactory interview presence'
        else:
            return 'Interview presence needs improvement'

    def _generate_feedback(self, content_score, delivery_score, communication_score, response_analysis, video_analysis):
        """Generate detailed feedback based on the evaluation."""
        feedback = []
        
        # Content feedback
        if content_score >= 0.8:
            feedback.append("Excellent response content with clear and relevant points.")
        elif content_score >= 0.6:
            feedback.append("Good response content, but could be more detailed or focused.")
        else:
            feedback.append("Response content needs improvement. Try to be more specific and relevant.")
            
        # Delivery feedback
        if delivery_score >= 0.8:
            feedback.append("Very professional delivery with strong presence.")
        elif delivery_score >= 0.6:
            feedback.append("Good delivery, but could improve engagement and confidence.")
        else:
            feedback.append("Work on delivery aspects like eye contact and engagement.")
            
        # Communication feedback
        if communication_score >= 0.8:
            feedback.append("Excellent communication skills demonstrated.")
        elif communication_score >= 0.6:
            feedback.append("Good communication, but watch for filler words and pacing.")
        else:
            feedback.append("Focus on improving clarity and reducing filler words.")
            
        return " ".join(feedback) 