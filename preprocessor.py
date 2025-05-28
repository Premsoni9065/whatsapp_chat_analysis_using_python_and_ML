import re
import pandas as pd
from datetime import datetime
from langdetect import detect, LangDetectException


def preprocess(data):
    # Pattern to match WhatsApp message format
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:AM|PM|am|pm)?)\s-\s([^:]+):\s(.+)'

    messages = re.findall(pattern, data)

    # If the first pattern fails (no AM/PM), try without it
    if not messages:
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)'
        messages = re.findall(pattern, data)

    df = pd.DataFrame(messages, columns=['date', 'user', 'message'])

    # Convert date strings to datetime objects
    date_formats = [
        '%m/%d/%y, %I:%M %p',  # 3/31/25, 15:54
        '%m/%d/%y, %H:%M',  # 3/31/25, 15:54 (24-hour format)
        '%d/%m/%y, %I:%M %p',  # 31/3/25, 15:54
        '%d/%m/%y, %H:%M'  # 31/3/25, 15:54 (24-hour format)
    ]

    for fmt in date_formats:
        try:
            df['date'] = df['date'].apply(lambda x: datetime.strptime(x, fmt))
            break
        except ValueError:
            continue

    # Extract additional datetime features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create time period
    df['period'] = df['hour'].apply(lambda x: f"{x}-{x + 1}")

    # Detect language
    def detect_language(text):
        try:
            return detect(text)
        except LangDetectException:
            return 'unknown'

    df['language'] = df['message'].apply(detect_language)

    return df