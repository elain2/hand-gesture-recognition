def main():
    frame_num = 0
    while True:
        print_num(frame_num)
        frame_num = frame_num + 1


def print_num(num):
    json_title = 'json/test_' + str(num).zfill(8) + '.json'
    print(json_title)


main()
