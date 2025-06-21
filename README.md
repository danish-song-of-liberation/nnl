# nnl 0.1.0 (W.I.P.)
Сommon lisp neural networks mvp framework

## Note

Please be aware: this code was formatted using Notepad++.  
Due to the way it handles indentation,
the indentation might look a bit unusual.  
Please don't be alarmed!**.**

## Note

This is an alpha version! API may change.

## Customizing the Calculus System

By default, `nnl` uses `single-float` for all numerical computations.  However, you can easily change the calculus system to use a different floating-point type, such as `double-float`, to increase precision.

To change the calculus system, use the `nnl.system:change-calculus-system!` function.  This function accepts a single argument: a symbol representing the desired floating-point type.

**Example:**

To switch to using `double-float`:

```lisp
(nnl.system:change-calculus-system! 'double-float)




















<h3 style="color: #5a1a8c; margin-top: 0;">Philipp Mainländer (1841–1876)</h3>

<blockquote style="font-style: italic; border-left: 3px solid #b57edc; padding-left: 15px; margin-left: 0;">
<p>Jede Handlung des Menschen, die höchste wie die niedrigste, ist egoistisch; denn sie entspringt einer bestimmten Individualität, einem bestimmten Ich, mit einem ausreichenden Motiv und kann in keiner Weise ausgelassen werden. Auf den Grund der Verschiedenheit der Charaktere einzugehen, ist hier nicht der richtige Ort; wir müssen es einfach als Tatsache akzeptieren.</p>

<p>Nun ist es für den barmherzigen Mann genauso unmöglich, seinen Nächsten hungern zu lassen, wie es für den hartherzigen Mann ist, den Armen zu helfen. Jeder der beiden handelt nach seinem Charakter, seiner Natur, seinem Ego, seinem Glück, folglich egoistisch; denn wenn der Barmherzige die Tränen anderer nicht trocknen würde, wäre er glücklich? Und wenn der Hartherzige das Leiden anderer linderte, wäre er zufrieden?</p>
</blockquote>

<div style="background-color: #e9d8fd; padding: 15px; border-radius: 5px; margin-top: 15px;">
<h4 style="color: #4b1a7a; margin-top: 0;">English Translation:</h4>

<blockquote style="font-style: normal; border-left: 3px solid #9f6ec1; padding-left: 15px;">
<p>Every action of man, the highest as well as the lowest, is egoistic; for it flows from a certain individuality, a certain I, with a sufficient motive, and can in no way be omitted.</p>

<p>To go into the reason of the difference of characters is not the place here; we have simply to accept it as a fact. Now it is just as impossible for the merciful man to let his neighbor starve as it is for the hard-hearted man to help the poor.</p>

<p>Each of the two acts according to his character, his nature, his ego, his happiness, consequently egoistically; for if the merciful one did not dry the tears of others, would he be happy? And if the hard-hearted one relieved the suffering of others, would he be satisfied?</p>
</blockquote>
</div>

</div>