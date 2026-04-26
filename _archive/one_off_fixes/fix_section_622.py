                with detail_col3:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"• Overall Grade: {signal['grade']}")
                    st.write(f"• Confidence: {signal['score']:.1f}/100")
                    st.write(f"• Recommendation: {signal['recommendation']}")
                
                # Key factors summary
                if signal.get('factors'):
                    st.markdown("---")
                    st.subheader("🎯 Key Decision Factors")
                    factors_text = " | ".join(signal['factors'][:3])  # Top 3 factors
                    st.info(f"**Primary factors:** {factors_text}")