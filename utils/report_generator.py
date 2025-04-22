from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import os
from typing import Dict

def generate_report(evaluation: Dict) -> str:
    """Generate a detailed PDF report from the evaluation data."""
    # Create report filename
    report_path = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        report_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8
    )
    normal_style = styles['Normal']
    
    # Build the document content
    story = []
    
    # Title
    story.append(Paragraph("AI Interview Evaluation Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Overall Score and Recommendation
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(f"Overall Score: {evaluation['final_score']}/10", subheading_style))
    story.append(Paragraph(f"Recommendation: {evaluation['recommendation']}", subheading_style))
    story.append(Spacer(1, 20))
    
    # Behavioral Assessment
    story.append(Paragraph("Behavioral Assessment", heading_style))
    behavioral = evaluation['behavioral_assessment']
    story.append(Paragraph(f"Score: {behavioral['score']}/10", normal_style))
    story.append(Paragraph(f"Attention Level: {behavioral['attention_level']}", normal_style))
    story.append(Paragraph(f"Eye Contact: {behavioral['eye_contact']:.1%}", normal_style))
    story.append(Paragraph(f"Body Language: {behavioral['body_language']}", normal_style))
    story.append(Spacer(1, 20))
    
    # Communication Assessment
    story.append(Paragraph("Communication Assessment", heading_style))
    communication = evaluation['communication_assessment']
    story.append(Paragraph(f"Score: {communication['score']}/10", normal_style))
    story.append(Paragraph(f"Clarity: {communication['clarity']}/10", normal_style))
    story.append(Paragraph(f"Confidence: {communication['confidence']:.1%}", normal_style))
    story.append(Paragraph(f"Overall Sentiment: {communication['sentiment']}", normal_style))
    story.append(Spacer(1, 20))
    
    # Content Assessment
    story.append(Paragraph("Content Assessment", heading_style))
    content = evaluation['content_assessment']
    
    # Create content scores table
    content_data = [
        ['Aspect', 'Score'],
        ['Relevance to Job', f"{content['relevance_score']}/10"],
        ['Technical Knowledge', f"{content['technical_score']}/10"],
        ['Problem Solving', f"{content['problem_solving_score']}/10"],
        ['Cultural Fit', f"{content['cultural_fit_score']}/10"]
    ]
    
    content_table = Table(content_data, colWidths=[300, 100])
    content_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(content_table)
    story.append(Spacer(1, 20))
    
    # Detailed Feedback
    story.append(Paragraph("Detailed Feedback", heading_style))
    feedback = evaluation['detailed_feedback']
    
    # Strengths
    story.append(Paragraph("Strengths:", subheading_style))
    for strength in feedback['strengths']:
        story.append(Paragraph(f"• {strength}", normal_style))
    story.append(Spacer(1, 10))
    
    # Areas for Improvement
    story.append(Paragraph("Areas for Improvement:", subheading_style))
    for improvement in feedback['areas_for_improvement']:
        story.append(Paragraph(f"• {improvement}", normal_style))
    story.append(Spacer(1, 10))
    
    # Interview Presence
    story.append(Paragraph("Overall Interview Presence:", subheading_style))
    story.append(Paragraph(feedback['interview_presence'], normal_style))
    
    # Build the PDF
    doc.build(story)
    
    return report_path 