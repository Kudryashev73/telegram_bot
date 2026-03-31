import logging
import asyncio
from io import BytesIO
from urllib.parse import quote
import os
import tempfile

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import LabeledPrice, PreCheckoutQuery, BufferedInputFile

from PIL import Image
import requests
from dotenv import load_dotenv

# ==================== ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ====================

# Загружаем токены из .env файла
load_dotenv()

# Получаем токены из переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Проверка что токены загружены
if not BOT_TOKEN:
    raise ValueError(
        "❌ BOT_TOKEN не найден!\n"
        "Создай файл .env и добавь:\n"
        "BOT_TOKEN=твой_токен_от_BotFather"
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("✅ Токены загружены из .env")


# ==================== НАСТРОЙКИ МОДЕЛЕЙ ====================

TEXT_TO_IMAGE_MODELS = {
    "flux_2_pro": {
        "id": "black-forest-labs/flux-2-pro",
        "name": "🔥 Flux 2 Pro",
        "description": "Новейшая модель Flux 2 (2025)",
        "cost": 2,
        "use_replicate": True,
        "replicate_model": "black-forest-labs/flux-2-pro"
    },
    "flux_schnell": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "name": "⚡ Flux Schnell",
        "description": "Быстрая генерация",
        "cost": 1,
        "use_replicate": False
    },
    "sdxl": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "🎨 SDXL",
        "description": "Stable Diffusion XL",
        "cost": 1,
        "use_replicate": False
    },
    "playground": {
        "id": "playgroundai/playground-v2.5-1024px-aesthetic",
        "name": "🎮 Playground",
        "description": "Красивые изображения",
        "cost": 1,
        "use_replicate": False
    }
}

IMAGE_TO_IMAGE_MODELS = {
    "flux_2_pro_img": {
        "id": "black-forest-labs/flux-2-pro",
        "name": "🔥 Flux 2 Pro",
        "description": "Flux 2 с входным изображением",
        "cost": 3,
        "replicate_model": "black-forest-labs/flux-2-pro"
    },
    "instruct_pix2pix": {
        "id": "timothybrooks/instruct-pix2pix",
        "name": "✨ InstructPix2Pix",
        "description": "Изменение по описанию",
        "cost": 2,
        "replicate_model": "timothybrooks/instruct-pix2pix"
    }
}

PRICES = {
    "5": 60,
    "10": 100,
    "20": 180
}


# ==================== ИНИЦИАЛИЗАЦИЯ ====================

storage = MemoryStorage()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=storage)

try:
    from huggingface_hub import InferenceClient
    hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None
    HF_AVAILABLE = bool(hf_client)
except:
    HF_AVAILABLE = False
    hf_client = None

try:
    import replicate
    REPLICATE_AVAILABLE = True
except:
    REPLICATE_AVAILABLE = False

user_balances = {}
user_photos = {}
user_text_model = {}
user_image_model = {}


class ImageGen(StatesGroup):
    text_to_image = State()
    image_upload = State()
    image_to_image = State()


# ==================== ФУНКЦИИ ГЕНЕРАЦИИ ====================
# (Оставляем без изменений - все функции generate_* такие же как были)

def generate_via_pollinations(prompt):
    try:
        logging.info(f"Pollinations: {prompt[:50]}")
        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        params = {"width": 1024, "height": 1024, "model": "flux", "nologo": "true"}
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 200:
            logging.info("Success")
            return BytesIO(response.content)
    except Exception as e:
        logging.error(f"Error: {e}")
    return None


def generate_via_replicate_text(prompt, model_id):
    if not REPLICATE_AVAILABLE or not REPLICATE_API_TOKEN:
        logging.error("Replicate not available")
        return None
    
    try:
        logging.info(f"Replicate Text-to-Image: {model_id}")
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        
        if "flux-2-pro" in model_id.lower():
            output = replicate.run(
                model_id,
                input={
                    "prompt": prompt,
                    "resolution": "1 MP",
                    "aspect_ratio": "1:1",
                    "output_format": "webp",
                    "output_quality": 80,
                    "safety_tolerance": 2
                }
            )
        else:
            output = replicate.run(model_id, input={"prompt": prompt})
        
        if output:
            url = output[0] if isinstance(output, list) else output
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                logging.info("Replicate Success")
                return BytesIO(response.content)
    
    except Exception as e:
        logging.error(f"Replicate error: {e}")
    
    return None


def generate_text_to_image(prompt, model_key):
    model_data = TEXT_TO_IMAGE_MODELS.get(model_key, {})
    model_id = model_data.get("id")
    use_replicate = model_data.get("use_replicate", False)
    
    if use_replicate:
        replicate_model = model_data.get("replicate_model", model_id)
        return generate_via_replicate_text(prompt, replicate_model)
    
    if hf_client:
        try:
            logging.info(f"HF: {model_id}")
            image = hf_client.text_to_image(prompt=prompt, model=model_id)
            if isinstance(image, Image.Image):
                buf = BytesIO()
                image.save(buf, format='PNG')
                buf.seek(0)
                logging.info("HF Success")
                return buf
            elif isinstance(image, bytes):
                logging.info("HF Success")
                return BytesIO(image)
        except Exception as e:
            logging.warning(f"HF failed: {e}")
    
    return generate_via_pollinations(prompt)


def generate_image_to_image(prompt, input_image, model_key):
    if not REPLICATE_AVAILABLE:
        logging.error("replicate not installed")
        return None
    
    if not REPLICATE_API_TOKEN:
        logging.error("REPLICATE_API_TOKEN not set")
        return None
    
    try:
        model_data = IMAGE_TO_IMAGE_MODELS.get(model_key, {})
        replicate_model = model_data.get("replicate_model", model_data.get("id"))
        
        logging.info(f"Replicate: {replicate_model}")
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        
        input_image.seek(0)
        pil_image = Image.open(input_image)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        if pil_image.width > 1024 or pil_image.height > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_path = temp_file.name
        temp_file.close()
        pil_image.save(temp_path, format='PNG')
        
        try:
            with open(temp_path, "rb") as f:
                if "flux-2-pro" in replicate_model.lower():
                    output = replicate.run(
                        replicate_model,
                        input={
                            "prompt": prompt,
                            "input_images": [f],
                            "resolution": "1 MP",
                            "aspect_ratio": "1:1",
                            "output_format": "webp",
                            "output_quality": 80,
                            "safety_tolerance": 2
                        }
                    )
                elif "instruct-pix2pix" in replicate_model.lower():
                    output = replicate.run(
                        replicate_model,
                        input={"image": f, "prompt": prompt, "num_inference_steps": 20}
                    )
                elif "sdxl" in replicate_model.lower():
                    output = replicate.run(
                        replicate_model,
                        input={"image": f, "prompt": prompt, "strength": 0.8}
                    )
                else:
                    output = replicate.run(replicate_model, input={"image": f, "prompt": prompt})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if output:
            url = output[0] if isinstance(output, list) else output
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                logging.info("Replicate Success")
                return BytesIO(response.content)
    
    except Exception as e:
        logging.error(f"Error: {e}")
    
    return None


def get_balance(user_id):
    return user_balances.get(user_id, 0)

def add_balance(user_id, amount):
    user_balances[user_id] = user_balances.get(user_id, 0) + amount

def use_balance(user_id, amount=1):
    if get_balance(user_id) >= amount:
        user_balances[user_id] -= amount
        return True
    return False


# ==================== HANDLERS ====================
# (Все handlers остаются без изменений)

@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    user_id = message.from_user.id
    
    if user_id not in user_balances:
        add_balance(user_id, 5)
        welcome = "🎉 Привет! Ты получил 5 бесплатных генераций!\n\n"
    else:
        welcome = "👋 С возвращением!\n\n"
    
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="✨ Создать из текста", callback_data="mode_text")],
        [types.InlineKeyboardButton(text="🖼 Изменить фото", callback_data="mode_image")],
        [types.InlineKeyboardButton(text="━━━━━━━━━━━━━━", callback_data="sep")],
        [types.InlineKeyboardButton(text="🎁 +5 бесплатно", callback_data="free")],
        [types.InlineKeyboardButton(text="💎 Купить", callback_data="prices")],
        [types.InlineKeyboardButton(text="📊 Баланс", callback_data="balance")]
    ])
    
    await message.answer(
        f"{welcome}🎨 AI Image Generator\n\n"
        f"🔥 Теперь с Flux 2 Pro!\n\n"
        f"💰 Баланс: {get_balance(user_id)} генераций\n\n"
        f"Выбери режим:",
        reply_markup=keyboard
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "📖 Инструкция\n\n"
        "Text-to-Image:\n"
        "1. Нажми Создать из текста\n"
        "2. Выбери модель\n"
        "3. Опиши картинку (English)\n\n"
        "Image-to-Image:\n"
        "1. Нажми Изменить фото\n"
        "2. Загрузи фото\n"
        "3. Опиши изменения (English)\n\n"
        "🔥 Flux 2 Pro - новейшая модель 2025\n\n"
        "Команды: /start /help /balance /cancel"
    )


@dp.message(Command("balance"))
async def cmd_balance(message: types.Message):
    await message.answer(f"💰 Баланс: {get_balance(message.from_user.id)} генераций")


@dp.message(Command("cancel"))
async def cmd_cancel(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("❌ Отменено\n/start")


@dp.callback_query(F.data == "sep")
async def cb_sep(callback: types.CallbackQuery):
    await callback.answer()


@dp.callback_query(F.data == "balance")
async def cb_balance(callback: types.CallbackQuery):
    await callback.answer(f"💰 {get_balance(callback.from_user.id)} генераций", show_alert=True)


@dp.callback_query(F.data == "free")
async def cb_free(callback: types.CallbackQuery):
    add_balance(callback.from_user.id, 5)
    await callback.answer("🎉 +5 генераций!", show_alert=True)
    await callback.message.answer(f"✅ Баланс: {get_balance(callback.from_user.id)}")


@dp.callback_query(F.data == "prices")
async def cb_prices(callback: types.CallbackQuery):
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="💎 5 ген. (~90₽)", callback_data="buy_5")],
        [types.InlineKeyboardButton(text="💎 10 ген. (~150₽)", callback_data="buy_10")],
        [types.InlineKeyboardButton(text="💎 20 ген. (~270₽)", callback_data="buy_20")],
        [types.InlineKeyboardButton(text="◀️ Назад", callback_data="back")]
    ])
    await callback.message.edit_text("💎 Купить генерации", reply_markup=keyboard)


@dp.callback_query(F.data == "back")
async def cb_back(callback: types.CallbackQuery):
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="✨ Создать из текста", callback_data="mode_text")],
        [types.InlineKeyboardButton(text="🖼 Изменить фото", callback_data="mode_image")],
        [types.InlineKeyboardButton(text="━━━━━━━━━━━━━━", callback_data="sep")],
        [types.InlineKeyboardButton(text="🎁 +5 бесплатно", callback_data="free")],
        [types.InlineKeyboardButton(text="💎 Купить", callback_data="prices")],
        [types.InlineKeyboardButton(text="📊 Баланс", callback_data="balance")]
    ])
    await callback.message.edit_text(
        f"🎨 AI Image Generator\n\n💰 Баланс: {get_balance(callback.from_user.id)}",
        reply_markup=keyboard
    )


@dp.callback_query(F.data.startswith("buy_"))
async def cb_buy(callback: types.CallbackQuery):
    count = callback.data.split("_")[1]
    await bot.send_invoice(
        chat_id=callback.from_user.id,
        title=f"{count} генераций",
        description="AI генерация картинок",
        payload=f"gens_{count}",
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label=f"{count} ген.", amount=PRICES[count])]
    )
    await callback.answer()


@dp.pre_checkout_query()
async def pre_checkout(pre_checkout: PreCheckoutQuery):
    await bot.answer_pre_checkout_query(pre_checkout.id, ok=True)


@dp.message(F.successful_payment)
async def payment_success(message: types.Message):
    count = int(message.successful_payment.invoice_payload.split("_")[1])
    add_balance(message.from_user.id, count)
    await message.answer(f"✅ Оплата прошла!\n💰 Баланс: {get_balance(message.from_user.id)}")


@dp.callback_query(F.data == "mode_text")
async def mode_text(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    
    if get_balance(callback.from_user.id) <= 0:
        await callback.message.answer("❌ Генерации закончились")
        return
    
    keyboard = []
    for key, model in TEXT_TO_IMAGE_MODELS.items():
        keyboard.append([types.InlineKeyboardButton(
            text=f"{model['name']} ({model['cost']})",
            callback_data=f"txt_{key}"
        )])
    keyboard.append([types.InlineKeyboardButton(text="◀️ Назад", callback_data="back")])
    
    await callback.message.edit_text(
        "✨ Text-to-Image\n\nВыбери модель:",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    )


@dp.callback_query(F.data.startswith("txt_"))
async def select_txt_model(callback: types.CallbackQuery, state: FSMContext):
    key = callback.data.replace("txt_", "")
    user_text_model[callback.from_user.id] = key
    model = TEXT_TO_IMAGE_MODELS[key]
    
    if model.get("use_replicate") and (not REPLICATE_AVAILABLE or not REPLICATE_API_TOKEN):
        await callback.answer("❌ Нужен Replicate API токен", show_alert=True)
        return
    
    await callback.answer(f"✅ {model['name']}", show_alert=True)
    await callback.message.edit_text(
        f"✅ {model['name']}\n\n"
        f"{model['description']}\n"
        f"Стоимость: {model['cost']} ген.\n\n"
        f"Отправь промпт (English):\n"
        f"Пример: cat in space suit\n\n"
        f"/cancel - отмена"
    )
    await state.set_state(ImageGen.text_to_image)


@dp.message(ImageGen.text_to_image, F.text)
async def process_txt_gen(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    key = user_text_model.get(user_id, "flux_schnell")
    model = TEXT_TO_IMAGE_MODELS[key]
    
    if get_balance(user_id) < model['cost']:
        await state.clear()
        await message.answer("❌ Недостаточно генераций")
        return
    
    prompt = message.text
    if len(prompt) < 3:
        await message.answer("⚠️ Промпт слишком короткий")
        return
    
    status = await message.answer("⏳ Генерирую...")
    
    try:
        img = generate_text_to_image(prompt, key)
        if img:
            await bot.send_photo(
                message.chat.id,
                BufferedInputFile(img.getvalue(), "img.png"),
                caption=f"✨ {prompt}\n🎨 {model['name']}\n💰 Осталось: {get_balance(user_id) - model['cost']}"
            )
            use_balance(user_id, model['cost'])
            await status.delete()
            
            keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
                [types.InlineKeyboardButton(text="🔄 Ещё", callback_data="mode_text")],
                [types.InlineKeyboardButton(text="🏠 Меню", callback_data="back")]
            ])
            await message.answer(f"✅ Готово! Баланс: {get_balance(user_id)}", reply_markup=keyboard)
        else:
            await status.edit_text("❌ Ошибка генерации")
    except Exception as e:
        logging.error(e)
        await status.edit_text("❌ Ошибка")
    finally:
        await state.clear()


@dp.callback_query(F.data == "mode_image")
async def mode_image(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    
    if get_balance(callback.from_user.id) <= 0:
        await callback.message.answer("❌ Генерации закончились")
        return
    
    if not REPLICATE_AVAILABLE or not REPLICATE_API_TOKEN:
        await callback.message.answer(
            "❌ Image-to-Image недоступен\n\n"
            "Нужен Replicate API токен\n"
            "https://replicate.com/account/api-tokens"
        )
        return
    
    keyboard = []
    for key, model in IMAGE_TO_IMAGE_MODELS.items():
        keyboard.append([types.InlineKeyboardButton(
            text=f"{model['name']} ({model['cost']})",
            callback_data=f"img_{key}"
        )])
    keyboard.append([types.InlineKeyboardButton(text="◀️ Назад", callback_data="back")])
    
    await callback.message.edit_text(
        "🖼 Image-to-Image\n\nВыбери модель:",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    )


@dp.callback_query(F.data.startswith("img_"))
async def select_img_model(callback: types.CallbackQuery, state: FSMContext):
    key = callback.data.replace("img_", "")
    user_image_model[callback.from_user.id] = key
    model = IMAGE_TO_IMAGE_MODELS[key]
    
    await callback.answer(f"✅ {model['name']}", show_alert=True)
    await callback.message.edit_text(
        f"✅ {model['name']}\n\n"
        f"{model['description']}\n"
        f"Стоимость: {model['cost']} ген.\n\n"
        f"Загрузи фото\n\n"
        f"/cancel - отмена"
    )
    await state.set_state(ImageGen.image_upload)


@dp.message(ImageGen.image_upload, F.photo)
async def upload_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"
    response = requests.get(url)
    user_photos[message.from_user.id] = BytesIO(response.content)
    
    await message.answer(
        "✅ Фото загружено!\n\n"
        "Опиши изменения (English):\n"
        "Пример: make it oil painting\n\n"
        "/cancel - отмена"
    )
    await state.set_state(ImageGen.image_to_image)


@dp.message(ImageGen.image_upload, ~F.photo)
async def wrong_upload(message: types.Message):
    await message.answer("⚠️ Отправь фото (не файл)")


@dp.message(ImageGen.image_to_image, F.text)
async def process_img_gen(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    key = user_image_model.get(user_id, "instruct_pix2pix")
    model = IMAGE_TO_IMAGE_MODELS[key]
    
    if get_balance(user_id) < model['cost']:
        await state.clear()
        await message.answer("❌ Недостаточно генераций")
        return
    
    if user_id not in user_photos:
        await state.clear()
        await message.answer("⚠️ Фото не найдено\n/start")
        return
    
    prompt = message.text
    if len(prompt) < 3:
        await message.answer("⚠️ Промпт слишком короткий")
        return
    
    status = await message.answer("⏳ Обрабатываю... (20-60 сек)")
    
    try:
        img = generate_image_to_image(prompt, user_photos[user_id], key)
        if img:
            await bot.send_photo(
                message.chat.id,
                BufferedInputFile(img.getvalue(), "result.png"),
                caption=f"🖼 {prompt}\n🎨 {model['name']}\n💰 Осталось: {get_balance(user_id) - model['cost']}"
            )
            use_balance(user_id, model['cost'])
            del user_photos[user_id]
            await status.delete()
            
            keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
                [types.InlineKeyboardButton(text="🔄 Ещё", callback_data="mode_image")],
                [types.InlineKeyboardButton(text="🏠 Меню", callback_data="back")]
            ])
            await message.answer(f"✅ Готово! Баланс: {get_balance(user_id)}", reply_markup=keyboard)
        else:
            await status.edit_text("❌ Ошибка обработки")
    except Exception as e:
        logging.error(e)
        await status.edit_text("❌ Ошибка")
    finally:
        await state.clear()


@dp.message(F.text)
async def other_text(message: types.Message):
    await message.answer("👋 Нажми /start")


# ==================== ЗАПУСК ====================

async def on_startup():
    logging.info("=" * 50)
    logging.info("🚀 Запуск бота...")
    info = await bot.get_me()
    logging.info(f"🤖 @{info.username}")
    logging.info(f"✅ HF: {HF_AVAILABLE}")
    
    if REPLICATE_AVAILABLE:
        logging.info("✅ Библиотека replicate установлена")
        if REPLICATE_API_TOKEN:
            logging.info(f"✅ REPLICATE_API_TOKEN: {REPLICATE_API_TOKEN[:10]}...")
            logging.info("✅ Image-to-Image доступен")
        else:
            logging.warning("⚠️ REPLICATE_API_TOKEN не установлен")
    else:
        logging.warning("⚠️ replicate не установлен")
    
    logging.info(f"🔥 Flux 2 Pro: {REPLICATE_AVAILABLE and bool(REPLICATE_API_TOKEN)}")
    logging.info(f"📦 Text: {len(TEXT_TO_IMAGE_MODELS)}")
    logging.info(f"📦 Image: {len(IMAGE_TO_IMAGE_MODELS)}")
    logging.info("=" * 50)


async def main():
    await on_startup()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("⚠️ Остановлено")