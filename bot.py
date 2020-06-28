import logging
import os

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
# from aiogram.dispatcher.filters.state import State, StatesGroup
import subprocess

API_TOKEN = '<>'

# Configure logging
logging.basicConfig(level = logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token = API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot,storage = storage)

# States
id = dict()

class person():
'''Создаем класс для хранения необходимой информации'''
    orders = 0

    def __init__(self, name, id):
        person.orders += 1
        self.id = str(id)
        self.name = name
        self.nst_anime = False
        self.photo_nst1 = None
        self.photo_nst2 = None
        self.photo_anime = None

    def sent(self, mode):
        '''Создаем ордер'''
        if mode == 'anime':
            result = self.id + ' ' + mode + ' ' + self.photo_anime + '\n'
            with open(f'order_{mode}.txt', 'a+') as f:
                f.write(result)
        else:
            result = self.id + ' ' + mode + ' ' + self.photo_nst1 +' ' + self.photo_nst2 + '\n'
            with open(f'order_{mode}.txt', 'a+') as f:
                f.write(result)

    def check(self):
        '''Очистка мусора'''
        if self.photo_nst1 != None:
            os.remove(self.photo_nst1)


markup = dict()
markup['start'] = types.ReplyKeyboardMarkup(resize_keyboard = True, selective = True, row_width = 3 )
markup['start'].add("Сделать stylе transfer", "Сдeлать anime сeлфи")
markup['start'].add("Нужна помощь?", )

markup['end'] = types.ReplyKeyboardMarkup(resize_keyboard = True, selective = True, row_width = 3)
markup['end'].add("Хотите начать заново? Нужна помощь?", )

@dp.message_handler(commands = 'start')
async def cmd_start(message: types.Message):
    """
    Conversation's entry point
    """
    # Set state
    print (message)
    id[message.from_user.id] = person(message.from_user.first_name, message.from_user.id)

    answer = f"Здравствуйте {message.from_user.first_name}!\nЧто вас интересует?\n" \
             f"Используйте кнопки!"

    await message.answer(answer,reply_markup = markup['start'])



@dp.message_handler(lambda message:
                    (message.text == '/help') | (message.text == 'Нужна помощь?') |
                    (message.text == "Хотите начать заново? Нужна помощь?"))

async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    # del file if is here
    if message.text == "Хотите начать заново? Нужна помощь?":
        id[message.from_user.id].check()



    result = '''Я могу сделать: 
            Style transfer
            Anime селфи
            
    Для того чтобы выбрать нажмите на соответствующию клопку '''

    id[message.from_user.id] = person(message.from_user.first_name, message.from_user.id)
    await message.answer(result, reply_markup = markup['start'])

@dp.message_handler(lambda message: message.text == "Сделать stylе transfer" )
async def echo(message: types.Message):

    id[message.from_user.id].nst_anime = 'nst'
    result = f'Отлично {id[message.from_user.id].name}! Пришлите мне две фотографии.' \
             f' Где первая фотография содержит контент, а вторая стиль.  ' \

    await message.answer(result, reply_markup = markup['end'])

@dp.message_handler(lambda message: message.text == "Сдeлать anime сeлфи" )
async def echo(message: types.Message):

    id[message.from_user.id].nst_anime = 'anime'
    result = f'Отлично {id[message.from_user.id].name}! Пришлите мне Selfie. ' \
             f'Данная возможность находится в стадии разработки и возможны непредвиденные результаты. ' \
             f'Чтобы минимизировать такие риска рекомендуем направлять фото только лица (как на документы)'
    # subprocess.Popen(['python3', 'send.py',]) # after all
    await message.answer(result, reply_markup = markup['end'])

@dp.message_handler(content_types = ['photo'])
async def handle_docs_photo(message):
    result_nst = f'Отлично {id[message.from_user.id].name}! Ты сделал все правильно. Я приступил к выполнению.' \
             f'Как задача будет выполнена я направлю тебе результат. ' \
                 f'Ожидаемое время выполнения 5 минут.'
    result_anime = f'Отлично {id[message.from_user.id].name}! Ты сделал все правильно. Я приступил к выполнению.' \
             f'Как задача будет выполнена я направлю тебе результат'
    idd = message.from_user.id
    if id[idd].nst_anime == 'anime':
        id[idd].photo_anime = message.photo[-1].file_id
        await message.photo[-1].download(id[idd].photo_anime)
        #check instance
        busy = os.path.exists('order_anime.txt')
        id[idd].sent('anime')
        id[idd] = person(message.from_user.first_name, message.from_user.id)
        await message.answer(result_anime, reply_markup = markup['start'])
        if not busy:
            subprocess.Popen(['python', 'send_anime.py', ])

    if id[idd].nst_anime == 'nst':
        if id[idd].photo_nst1:
            id[idd].photo_nst2 = message.photo[-1].file_id
            await message.photo[-1].download(id[idd].photo_nst2)
            # check instance
            busy = os.path.exists('order_nst.txt')
            id[idd].sent('nst')
            id[idd] = person(message.from_user.first_name, message.from_user.id)
            await message.answer(result_nst, reply_markup = markup['start'])
            if not busy:
                subprocess.Popen(['python', 'send_nst.py', ])
        else:
            id[idd].photo_nst1 = message.photo[-1].file_id
            await message.photo[-1].download(id[idd].photo_nst1)
            await message.answer('Жду второе фото...',)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates = True)
