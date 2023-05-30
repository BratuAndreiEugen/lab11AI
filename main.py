from iteratia1.Emoji import emoji
from iteratia2.EmotionsPreTrained import preTrained, ferPreTrained
from iteratia3.Emotions import emotions, emotionsAutomatedExtract
from iteratia3.EmotionsMultiLabel import multiLabel

while 1:
    print("\nOptiuni")
    print("1. Clasificare emoji")
    print("2. Pre-antrenat pentru clasificare emotii")
    print("3. Antrenare model pentru clasificare emotii")
    print("4. Model pentru clasificare emotii ( extragere automata a feature-urilor )")
    print("5. Model pentru multi-label")
    i = input("Alege : ")
    if i == "1":
        emoji()

    if i == "2":
        preTrained()

    if i == "3":
        emotions()

    if i == "4":
        emotionsAutomatedExtract()

    if i == "5":
        multiLabel()

    if i == "exit" or i == "0":
        exit(0)