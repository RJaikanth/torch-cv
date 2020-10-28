TEMP = "/tmp/temp.yml"


def create_empty_config():
    with open(TEMP, "w") as f:
        pass


def create_none_config():
    data = """
    field0: !none
    """
    with open(TEMP, "w") as f:
        f.write(data)


def create_join_config():
    data = """
    field0: !join ["a", "b", "c"]
    """
    with open(TEMP, "w") as f:
        f.write(data)


if __name__ == '__main__':
    create_none_config()
