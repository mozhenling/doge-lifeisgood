#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'update algorithms'

git push origin master

echo '------- update complete --------'