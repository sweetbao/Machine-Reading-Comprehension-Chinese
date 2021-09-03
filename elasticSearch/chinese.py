han_list = ["零" , "一" , "二" , "三" , "四" , "五" , "六" , "七" , "八" , "九"]
unit_list = ["","","十" , "百" , "千"]

def four_to_han(num_str):
    num_str = str(num_str)
    result = ""
    num_len = len(num_str)
    for i in range(num_len):
        num = int(num_str[i])
        if i!=num_len-1:
            if num!=0:
                result=result+han_list[num]+unit_list[num_len-i]
            else:
                if result[-1]=='零':
                    continue
                else:
                    result=result+'零'
        else:
            if num!=0:
                result += han_list[num]

    return result
if __name__ == '__main__':
    print (four_to_han("290"))