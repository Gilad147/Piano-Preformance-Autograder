\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % Thompson’s modern course, p. 4 

\new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \key c \major
  \time 4/4

	e2 f | g c, | d f | e1 | e2 f | g c, | d e | c1 \bar "|."
}

\new Staff = "lower" 
\relative c {
  \clef bass
  \key c \major
  \time 4/4

    c1 | e | g | c, | c | e | f2 g | e1 \bar "|."
}
>>

}
\layout {
  \context {
      \Score 
      proportionalNotationDuration =  #(ly:make-moment 1/5)
 }
 }
\midi {\tempo 4 = 60}
}
