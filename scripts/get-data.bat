@echo off
if not exist .\data (
    kaggle competitions download -c airbnb-recruiting-new-user-bookings
    mkdir .\data
    tar -xf airbnb-recruiting-new-user-bookings.zip -C .\data
    if not exist .\data\csv (
        mkdir .\data\csv
    )
    for %%f in (.\data\*.zip) do (
        tar -xf %%f -C .\data\csv
        del %%f
    )
    del airbnb-recruiting-new-user-bookings.zip
)
