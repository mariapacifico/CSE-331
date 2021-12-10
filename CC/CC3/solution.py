"""
CC3 Student Submission
Name:
"""

from typing import List, Tuple


def calculate(participants: List[str], character_details: List[Tuple[str, str, int]]) \
        -> List[Tuple[str, int]]:
    """
    Calculates the scores of each character after fighting
    :param participants: Names of all of the characters in Japanese or English
    :param character_details: List that contains the Japenese and English name and the score of the character
    """

    # 1. change names in list to english to make easier to index later

    # create dictionary for jap to english names : dict_jap_names[jap name] = english name
    dict_jap_names = {}

    for i in character_details:
        # the first index is the english name
        if i[0][0].isascii():
            dict_jap_names[i[1]] = i[0]

        # first index is the japaneese name
        else:
            dict_jap_names[i[0]] = i[1]

    # create dictionary for english to jap names : dict_jap_names[eng name] = jap nam
    # used later if need to change back to jap
    dict_eng_names = {}

    for i in character_details:
        # the first index is the english name
        if i[0][0].isascii():
            dict_eng_names[i[0]] = i[1]

        # is the japaneese name
        else:
            dict_eng_names[i[1]] = i[0]

    # iterate through list to check, change to english names
    for i in range(len(participants)):
        # the name is jap
        if participants[i] in dict_jap_names:
            participants[i] = dict_jap_names[participants[i]] # get the english name from the dict, change in the list


    # 2. create dictionary with the points : dict_points[english name] = points
    dict_points = {}

    for i in character_details:
        # jap name
        if i[0] in dict_jap_names:
            dict_points[i[1]] = i[2]

        # is the english name
        else:
            dict_points[i[0]] = i[2]

    # 3. get the scores

    final_scores = []

    # add names to final
    for i in participants:
        # add jap names if score is larger than 9000
        if dict_points[i] > 9000:
            final_scores.append((dict_eng_names[i], 0))

        else:
            final_scores.append((i, 0))

    # use stack to calculate scores
    stack = []
    for index in range(len(participants)):

        # add to stack if empty
        if stack == []:

            #stack[][0] = index of the name
            #stack[][1] = track the score
            stack.append((index, 0))
            continue

        # add to stack if the current index's score is less than or equal to
        # the last value in the stack
        stacks_score = dict_points[participants[stack[-1][0]]]
        indexs_score = dict_points[participants[index]]

        if stacks_score >= indexs_score:
            stack.append((index, 0))
            continue

        # when the index's score is less than the stack:
        # iterate through stack, update scores of stack and in final list
        while stack:
            top = stack.pop()
            # update score in final list
            final_scores[top[0]] = (final_scores[top[0]][0], top[1])

            # update the next value in stack
            if stack != []:
                # update stack value by adding the score of the popped value and the score
                # of the popped character
                total = stack[-1][1] + dict_points[participants[top[0]]] + top[1]
                stack[-1] = (stack[-1][0], total)

                # when the next score less than what's in stack, need to add to stack
                # do this, by exiting while loop and then adding index to stack
                if dict_points[participants[stack[-1][0]]] >= indexs_score:
                    break

        stack.append((index, 0))

    # need to account for values in stack
    while stack:
        # add the top values to final
        top = stack.pop()
        final_scores[top[0]] = (final_scores[top[0]][0], top[1])

        # when there are still values, need to update the scores
        if stack != []:
            total = stack[-1][1] + dict_points[participants[top[0]]] + top[1]
            stack[-1] = (stack[-1][0], total)

    return final_scores



