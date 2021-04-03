black:
	python -m black .

gitall:
	git add .
	git commit -m $$me
	git push

try:
	@echo $$FOO
