# Not Organized data like a pdf which have code or we have some different type of data like we have in our github readme file (markdown text)
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyMuPDFLoader

# loader=PyMuPDFLoader('speech.pdf')
# docs=loader.load()

text="""
    class Animal:
    def __init__(self, name, age):
        self._name = name        # protected attribute
        self._age = age

    def speak(self):
        return "This animal makes a sound."

    def info(self):
        return f"Name: {self._name}, Age: {self._age}"

class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

    def speak(self):a
        return "Woof! Woof!"

    def info(self):
        return f"Dog -> {super().info()}, Breed: {self.breed}"

class Cat(Animal):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color

    def speak(self):
        return "Meow!"

    def info(self):
        return f"Cat -> {super().info()}, Color: {self.color}"

# Polymorphism demonstration
def animal_sound(animal: Animal):
    print(animal.info())
    print("Sound:", animal.speak(), "\n")

# Create objects
dog = Dog("Buddy", 3, "Golden Retriever")
cat = Cat("Whiskers", 2, "White")

# Call the function for both animals
animal_sound(dog)
animal_sound(cat)
"""
splitter=RecursiveCharacterTextSplitter.from_language(
    # You can also Use for Different Languages like java,Js,php,markdown etcc
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

# result=splitter.split_documents(docs)
result=splitter.split_text(text)

print(result[3])