def meowFunc(**kwargs):
    options = {"option1": "default_value1", "option2": "default_value2", "option3": "default_value3"}

    options = kwargs.pop("option21")
    print("opts", options)
    print("kw", kwargs)


meowFunc(option1="new_value1", option3="new_value3")


# testFunc(option2="new_value2")
