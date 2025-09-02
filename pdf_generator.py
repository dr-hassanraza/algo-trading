#!/usr/bin/env python3
"""
PDF Report Generator for PSX Trading Bot
========================================
Generates professional PDF reports for trading analysis
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from datetime import datetime
import io
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

class PDFReportGenerator:
    """Generate professional PDF reports for trading analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        styles = {}
        
        # Title style
        styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        # Subtitle style
        styles['CustomSubtitle'] = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.darkgreen,
            spaceAfter=20
        )
        
        # Header style
        styles['CustomHeader'] = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.darkred,
            spaceAfter=15
        )
        
        return styles
    
    def generate_analysis_report(self, symbol, analysis_data, chart_fig=None):
        """Generate PDF report for stock analysis"""
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph(f"PSX Trading Analysis Report - {symbol}", self.custom_styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = Paragraph(f"<b>Generated:</b> {report_date}<br/><b>Symbol:</b> {symbol}<br/><b>Market:</b> Pakistan Stock Exchange", self.styles['Normal'])
        story.append(metadata)
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['CustomSubtitle']))
        if 'recommendation' in analysis_data:
            summary_text = f"""
            <b>Recommendation:</b> {analysis_data.get('recommendation', 'N/A')}<br/>
            <b>Signal Strength:</b> {analysis_data.get('signal_strength', 'N/A')}<br/>
            <b>Risk Level:</b> {analysis_data.get('risk_level', 'N/A')}<br/>
            <b>Target Price:</b> {analysis_data.get('target_price', 'N/A')}<br/>
            """
            story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Technical Analysis
        story.append(Paragraph("Technical Analysis", self.custom_styles['CustomSubtitle']))
        
        if 'technical_indicators' in analysis_data:
            # Create table for technical indicators
            tech_data = analysis_data['technical_indicators']
            table_data = [['Indicator', 'Value', 'Signal']]
            
            for indicator, values in tech_data.items():
                if isinstance(values, dict):
                    value = values.get('value', 'N/A')
                    signal = values.get('signal', 'N/A')
                else:
                    value = str(values)
                    signal = 'N/A'
                table_data.append([indicator.replace('_', ' ').title(), str(value), str(signal)])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Chart (if provided)
        if chart_fig:
            story.append(Paragraph("Price Chart", self.custom_styles['CustomSubtitle']))
            # Convert Plotly figure to image
            img_buffer = self._plotly_to_image(chart_fig)
            if img_buffer:
                img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
        
        # Market Data
        if 'market_data' in analysis_data:
            story.append(Paragraph("Market Data", self.custom_styles['CustomSubtitle']))
            market_data = analysis_data['market_data']
            market_text = f"""
            <b>Current Price:</b> {market_data.get('current_price', 'N/A')}<br/>
            <b>Daily Change:</b> {market_data.get('change', 'N/A')} ({market_data.get('change_percent', 'N/A')}%)<br/>
            <b>Volume:</b> {market_data.get('volume', 'N/A')}<br/>
            <b>52 Week High:</b> {market_data.get('high_52w', 'N/A')}<br/>
            <b>52 Week Low:</b> {market_data.get('low_52w', 'N/A')}<br/>
            """
            story.append(Paragraph(market_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Risk Analysis
        if 'risk_analysis' in analysis_data:
            story.append(Paragraph("Risk Analysis", self.custom_styles['CustomSubtitle']))
            risk_data = analysis_data['risk_analysis']
            risk_text = f"""
            <b>Volatility:</b> {risk_data.get('volatility', 'N/A')}<br/>
            <b>Beta:</b> {risk_data.get('beta', 'N/A')}<br/>
            <b>Risk Score:</b> {risk_data.get('risk_score', 'N/A')}<br/>
            <b>Stop Loss:</b> {risk_data.get('stop_loss', 'N/A')}<br/>
            """
            story.append(Paragraph(risk_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Footer
        footer = Paragraph(
            "This report is generated by PSX Trading Bot for informational purposes only. "
            "Please consult with a financial advisor before making investment decisions.",
            self.styles['Italic']
        )
        story.append(Spacer(1, 40))
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _plotly_to_image(self, fig):
        """Convert Plotly figure to image buffer"""
        try:
            img_bytes = pio.to_image(fig, format="png", width=600, height=400)
            return io.BytesIO(img_bytes)
        except Exception as e:
            print(f"Error converting chart to image: {e}")
            return None
    
    def generate_portfolio_report(self, portfolio_data):
        """Generate PDF report for portfolio"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph("PSX Portfolio Report", self.custom_styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = Paragraph(f"<b>Generated:</b> {report_date}<br/><b>Market:</b> Pakistan Stock Exchange", self.styles['Normal'])
        story.append(metadata)
        story.append(Spacer(1, 30))
        
        # Portfolio Summary
        story.append(Paragraph("Portfolio Summary", self.custom_styles['CustomSubtitle']))
        
        if 'holdings' in portfolio_data:
            holdings_data = [['Symbol', 'Shares', 'Current Price', 'Market Value', 'P&L']]
            
            for holding in portfolio_data['holdings']:
                holdings_data.append([
                    holding.get('symbol', 'N/A'),
                    str(holding.get('shares', 'N/A')),
                    str(holding.get('current_price', 'N/A')),
                    str(holding.get('market_value', 'N/A')),
                    str(holding.get('pnl', 'N/A'))
                ])
            
            table = Table(holdings_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

def create_download_link(buffer, filename):
    """Create a download link for PDF"""
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF Report</a>'
    return href