###Overview

Have you ever wondered if different music genres use certain words more often than others? Do country songs talk about dirt roads and beer more often than rock songs? Do rap songs swear more often than jazz? If you're like [Alex](https://www.linkedin.com/in/alexanderranney) and I, these questions keep you up at night. Created as a CS capstone project, this project set out to answer those questions and more. Along the way, we learned more about ourselves than we originally intentioned (and certainly less about lyrics, the project wasn't that successful).

Our project created a corpus with around ~2,000 songs along with their genres and lyrics. We implemented k nearest neighbors and Naive Bayes algorithms to predict a song's genre. Our story is as follows.

###The Corpus

Building a database of songs with categories and lyrics is quite difficult, for three main reasons:

1. **Categories** - Listeners have a hard time agreeing on categories for songs, especially for the oh-so-popular crossover hits. Is Taylor Swift's _Red_ country or pop? What's the difference between Blink-182 rock and The Rolling Stones rock? 
2. **Lyrics** - Music studios have forbidden consumption of lyrics through API's, leaving only large databases at your disposal. Or scraping. Lots of scraping.
3. **Cohesion** - While it's not impossible to find a list of songs, a list of lyrics, or a list of categories, it's impossible to find a database with all three neatly lined up. In our experience, you need to pick one and then cross reference the others (vlookup anyone? or perhaps a JOIN?)

So what did we do? Enter [Dave Tompkins](https://cs.uwaterloo.ca/~dtompkin/), a CS professor at the University of Waterloo. His wonderful database provided us with a large enough corpus with not only song titles, but also artists, categories and release dates. His ~2,000 songs provided us with the fuel we needed to complete this project, so here's to you Dave! (There is an alternative database called the [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/), but we couldn't get it working.)

We generated a list of songs, sent them off to LyricWiki and scraped the results page to pull together the lyrics. Save it in a database and you're good to go.

###The Machine Learning

We implemented a basic Naive Bayes and k-nearest neighbors algorithm and tested each of them separately against the built-in sci-kit algorithms.