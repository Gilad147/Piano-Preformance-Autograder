\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % Alfred’s p. 17 Jingle Bells

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \key c \major
  \time 4/4

  e4 e e2 | e4 e e2 | e4 g c, d | e1 | f4 f f f | f e e e | e d d e | d2 g |e4 e e2 | e4 e e2 | e4 g c, d | e1 | f4 f f f | f e e e | g g f d | c1   \bar "|."
}

\new Staff = "lower" 
\relative c {
  \clef bass
  \key c \major
  \time 4/4

  <c g'>1 | <c g'> | <c g'> | <c g'> | <d g> | <c g'> | d | g | <c, g'> | <c g'> | <c g'> | <c g'> | <d g> | <c g'> | <d g>2 g | <c, g'>1 \bar "|."
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
