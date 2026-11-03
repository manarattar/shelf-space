# What this tool says, in plain language

**The question.** A shelf has a fixed length. Which products should get more of it, and which less?

**What the tool does, in three steps.**

1. **Predicts what each product will sell** over the next four weeks. It looks at recent sales, the
   price and the calendar (weekdays, holidays). It is checked on four weeks of sales it never saw,
   and compared with a simple rule of thumb: "same as the last four same weekdays".
   On the demo data, the prediction of four-week sales per product is about 9% closer to reality
   than that rule of thumb.
2. **Moves space to where it earns most.** Extra space for a product helps less and less (the
   fifth facing sells less than the second). The tool gives the next facing to whichever product
   gains most from it, until the shelf is full.
3. **Shows how you would find out if it works.** Before changing real shelves you run a store
   test. The tool simulates one: how many stores you need, and how big a change you could notice.

**What to take from the demo numbers.**

- Against an *equal split* (every product the same space), the recommended shelf earns 10 to 26%
  more in the model. Most of that comes from simply not splitting space equally.
- Against a *sensible rule* (space follows expected revenue), the gain is below 1%. A change that
  small needs roughly 900 to 1,000 stores per group to prove in a test. With 30 stores per group you could
  only notice a change above about 4%.

**What the demo does not know.** It uses practice data, not Albert Heijn data. How strongly sales
react to extra shelf space is an assumption in the model, not a measurement. Real shelf plans,
margins, product substitution and supplier agreements are not included. The method carries over;
the numbers do not.
