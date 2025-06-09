#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'accepted'

git push origin master

echo '------- update complete --------'