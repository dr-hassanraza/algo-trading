from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import re
import pandas as pd
from io import StringIO

app = FastAPI()

# Allow all origins for development purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_hitlist(markdown_content):
    """
    Parses the hitlist markdown content to extract buy and sell opportunities.
    """
    buy_opportunities = []
    sell_opportunities = []

    # Regex to find the sections
    buy_section_match = re.search(r"📈 TOP BUY OPPORTUNITIES (\d+ signals):\s*-+\s*([\s\S]*?)(?=\n\n\S|\Z)", markdown_content)
    sell_section_match = re.search(r"📉 TOP SELL OPPORTUNITIES (\d+ signals):\s*-+\s*([\s\S]*?)(?=\n\n\S|\Z)", markdown_content)

    if buy_section_match:
        buy_section = buy_section_match.group(1)
        # Split into individual stock entries
        buy_entries = buy_section.strip().split('\n\n')
        for entry in buy_entries:
            lines = entry.strip().split('\n')
            if len(lines) >= 3:
                # Extracting details from the first line
                main_line = lines[0]
                match = re.match(r'\s*\d+\.\s+([\w&]+)\s+BUY\s+([\d\.]+)\%\s+Entry:\s+([\d\.]+)\s+SL:\s+([\d\.]+)\s+TP:\s+([\d\.]+)', main_line)
                if match:
                    symbol, confidence, entry_price, sl, tp = match.groups()
                    
                    # Extracting details from the second line (Pos, Vol, R/R, etc.)
                    pos_line = lines[1]
                    pos_match = re.search(r'Pos:\s+([\d\.]+)%', pos_line)
                    pos = pos_match.group(1) if pos_match else 'N/A'
                    
                    vol_match = re.search(r'Vol:(\S+)', pos_line)
                    vol = vol_match.group(1) if vol_match else 'N/A'
                    
                    rr_match = re.search(r'R/R:\s+([\d\.]+)', pos_line)
                    rr = rr_match.group(1) if rr_match else 'N/A'
                    
                    # Extracting reason from the third line
                    reason = lines[2].replace('Reason:', '').strip()
                    
                    buy_opportunities.append({
                        "symbol": symbol,
                        "signal": "BUY",
                        "confidence": float(confidence),
                        "entry": float(entry_price),
                        "stop_loss": float(sl),
                        "take_profit": float(tp),
                        "position_size": pos,
                        "volume": vol,
                        "risk_reward": rr,
                        "reason": reason
                    })


    if sell_section_match:
        sell_section = sell_section_match.group(1)
        # Split into individual stock entries
        sell_entries = sell_section.strip().split('\n\n')
        for entry in sell_entries:
            lines = entry.strip().split('\n')
            if len(lines) >= 3:
                main_line = lines[0]
                match = re.match(r'\s*\d+\.\s+([\w&]+)\s+SELL\s+([\d\.]+)\%\s+Entry:\s+([\d\.]+)\s+SL:\s+([\d\.]+)\s+TP:\s+([\d\.]+)', main_line)

                if match:
                    symbol, confidence, entry_price, sl, tp = match.groups()

                    # Extracting details from the second line
                    pos_line = lines[1]
                    pos_match = re.search(r'Pos:\s+([\d\.]+)%', pos_line)
                    pos = pos_match.group(1) if pos_match else 'N/A'

                    vol_match = re.search(r'Vol:(\S+)', pos_line)
                    vol = vol_match.group(1) if vol_match else 'N/A'

                    rr_match = re.search(r'R/R:\s+([\d\.]+)', pos_line)
                    rr = rr_match.group(1) if rr_match else 'N/A'
                    
                    # Extracting reason from the third line
                    reason = lines[2].replace('Reason:', '').strip()

                    sell_opportunities.append({
                        "symbol": symbol,
                        "signal": "SELL",
                        "confidence": float(confidence),
                        "entry": float(entry_price),
                        "stop_loss": float(sl),
                        "take_profit": float(tp),
                        "position_size": pos,
                        "volume": vol,
                        "risk_reward": rr,
                        "reason": reason
                    })

    return {"buy_opportunities": buy_opportunities, "sell_opportunities": sell_opportunities}

@app.get("/api/hitlist")
def get_hitlist():
    try:
        with open("PSX_HITLIST_CURRENT.md", "r") as f:
            content = f.read()
        
        hitlist_data = parse_hitlist(content)
        return hitlist_data
        
    except FileNotFoundError:
        return {"error": "Hitlist file not found."}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
