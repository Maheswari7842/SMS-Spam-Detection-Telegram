import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import joblib

# Load your trained model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the /start command is issued."""
    await update.message.reply_text('Hello! I am your Spam Detection Bot. Send me a message, and I will tell you if it is spam or not.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user messages and classify them as spam or not."""
    user_message = update.message.text
    user_message_vectorized = vectorizer.transform([user_message])
    prediction = model.predict(user_message_vectorized)[0]

    if prediction == 1:
        response = "This message is detected as spam."
    else:
        response = "This message is not spam."

    await update.message.reply_text(response)

def main() -> None:
    """Start the bot."""
    # Replace 'YOUR_API_TOKEN' with your actual bot token from BotFather
    API_TOKEN = '7633683466:AAHf-1TshEtlOR3WNl5aGi13awP4T9K3amQ'

    # Create the Application and pass it your bot's token
    app = Application.builder().token(API_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until you send a signal to stop
    app.run_polling()

if __name__ == '__main__':
    main()